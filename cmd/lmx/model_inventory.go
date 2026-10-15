package main

import (
	"bufio"
	"encoding/binary"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"math"
	"os"
	"path/filepath"
	"regexp"
	"sort"
	"strings"
	"text/tabwriter"
)

// `lmx model list` / `lmx model show`: one inventory of every local model,
// whether lmx registered it (single files and snapshots), the Hugging Face
// cache holds it, or it sits in a plain model directory (--scan-dir /
// LMX_MODEL_DIRS). Details come from the files themselves: config.json,
// safetensors headers and GGUF metadata. Nothing is downloaded or executed.

type modelInventoryFile struct {
	Name string `json:"name"`
	Size int64  `json:"size"`
}

type modelInventoryEntry struct {
	ID                 string               `json:"id"`
	Kind               string               `json:"kind"` // file | snapshot | hf-cache | directory
	Source             string               `json:"source"`
	Name               string               `json:"name"`
	Repo               string               `json:"repo,omitempty"`
	Revision           string               `json:"revision,omitempty"`
	File               string               `json:"file,omitempty"`
	Path               string               `json:"path"`
	Status             string               `json:"status"`
	SourceError        string               `json:"sourceError,omitempty"`
	Size               int64                `json:"size"`
	FileCount          int                  `json:"fileCount"`
	Format             string               `json:"format,omitempty"`
	Quantization       string               `json:"quantization,omitempty"`
	QuantizationSource string               `json:"quantizationSource,omitempty"`
	Architecture       string               `json:"architecture,omitempty"`
	ModelType          string               `json:"modelType,omitempty"`
	Parameters         int64                `json:"parameters,omitempty"`
	SizeLabel          string               `json:"sizeLabel,omitempty"`
	ContextLength      int64                `json:"contextLength,omitempty"`
	Layers             int64                `json:"layers,omitempty"`
	Experts            int64                `json:"experts,omitempty"`
	ExpertsPerToken    int64                `json:"expertsPerToken,omitempty"`
	LoadedAt           []string             `json:"loadedAt,omitempty"` // base URLs of lmx model load servers
	SHA256             string               `json:"sha256,omitempty"`
	Managed            bool                 `json:"managed,omitempty"`
	Integrity          string               `json:"integrity,omitempty"`
	ExpectedHash       string               `json:"expectedHash,omitempty"`
	UpstreamVerified   bool                 `json:"upstreamVerified,omitempty"`
	Files              []modelInventoryFile `json:"files,omitempty"`
}

var (
	ggufSplitPattern  = regexp.MustCompile(`-(\d{5})-of-(\d{5})\.gguf$`)
	weightExtensions  = map[string]string{".safetensors": "safetensors", ".bin": "pytorch", ".pt": "pytorch", ".pth": "pytorch"}
	ggufFileTypeNames = map[uint32]string{
		0: "F32", 1: "F16", 2: "Q4_0", 3: "Q4_1", 7: "Q8_0", 8: "Q5_0", 9: "Q5_1", 10: "Q2_K",
		11: "Q3_K_S", 12: "Q3_K_M", 13: "Q3_K_L", 14: "Q4_K_S", 15: "Q4_K_M", 16: "Q5_K_S",
		17: "Q5_K_M", 18: "Q6_K", 19: "IQ2_XXS", 20: "IQ2_XS", 21: "Q2_K_S", 22: "IQ3_XS",
		23: "IQ3_XXS", 24: "IQ1_S", 25: "IQ4_NL", 26: "IQ3_S", 27: "IQ3_M", 28: "IQ2_S",
		29: "IQ2_M", 30: "IQ4_XS", 31: "IQ1_M", 32: "BF16", 36: "TQ1_0", 37: "TQ2_0", 38: "MXFP4",
	}
)

// modelInventory gathers every source. Unreadable sources become entries with a
// status, never a failed listing.
func modelInventory(args cliArgs) (string, []modelInventoryEntry, error) {
	root, err := modelStoreRoot(args)
	if err != nil {
		return "", nil, err
	}
	entries, err := registeredModelEntries(root)
	if err != nil {
		return "", nil, err
	}
	snapshots, err := modelSnapshotInventory(root)
	if err != nil {
		return "", nil, err
	}
	for _, snapshot := range snapshots {
		entries = append(entries, snapshotEntry(snapshot))
	}
	if !hasFlag(args, "no-hf-cache") {
		entries = append(entries, huggingFaceCacheEntries()...)
	}
	for _, dir := range modelScanDirs(args) {
		entries = append(entries, scanDirEntries(dir)...)
	}
	markModelUsage(entries)
	sort.SliceStable(entries, func(i, j int) bool {
		if !strings.EqualFold(entries[i].Name, entries[j].Name) {
			return strings.ToLower(entries[i].Name) < strings.ToLower(entries[j].Name)
		}
		return entries[i].Kind < entries[j].Kind
	})
	return root, entries, nil
}

func registeredModelEntries(root string) ([]modelInventoryEntry, error) {
	dir, err := modelStoreCheckedPath(root, "registrations")
	if err != nil {
		return nil, err
	}
	names, err := os.ReadDir(dir)
	if err != nil && !errors.Is(err, os.ErrNotExist) {
		return nil, err
	}
	entries := []modelInventoryEntry{}
	for _, name := range names {
		if !strings.HasSuffix(name.Name(), ".json") {
			continue
		}
		artifact, err := readModelArtifact(root, strings.TrimSuffix(name.Name(), ".json"))
		if err != nil {
			return nil, fmt.Errorf("read model registration %s: %w", name.Name(), err)
		}
		entry := modelInventoryEntry{
			ID: artifact.ID, Kind: "file", Name: firstNonEmpty(artifact.Repo, artifact.File), Repo: artifact.Repo,
			Revision: artifact.Revision, File: artifact.File, Path: artifact.Path, Status: "available",
			Size: artifact.Size, FileCount: 1, SHA256: artifact.SHA256, Managed: artifact.Managed,
			Integrity: artifact.Integrity, ExpectedHash: artifact.ExpectedHash, UpstreamVerified: artifact.UpstreamVerified,
			Files: []modelInventoryFile{{Name: artifact.File, Size: artifact.Size}},
		}
		switch {
		case artifact.Managed:
			entry.Source = "managed"
		case artifact.Repo != "":
			entry.Source = "hf-cache-reference"
		default:
			entry.Source = "imported"
		}
		info, err := os.Stat(artifact.Path)
		switch {
		case errors.Is(err, os.ErrNotExist):
			entry.Status = "missing"
		case err != nil:
			entry.Status, entry.SourceError = "unreadable", err.Error()
		case !info.Mode().IsRegular():
			entry.Status = "not-regular"
		default:
			describeWeightFile(&entry, artifact.Path)
		}
		entries = append(entries, entry)
	}
	return entries, nil
}

func snapshotEntry(snapshot modelSnapshot) modelInventoryEntry {
	entry := modelInventoryEntry{
		ID: snapshot.ID, Kind: "snapshot", Source: "lmx-snapshot", Name: snapshot.Repo, Repo: snapshot.Repo,
		Revision: snapshot.Revision, Path: snapshot.Path, Status: snapshot.State, Size: snapshot.Size, FileCount: len(snapshot.Files),
	}
	for _, file := range snapshot.Files {
		entry.Files = append(entry.Files, modelInventoryFile{Name: file.File, Size: file.Size})
	}
	if snapshot.State == "ready" {
		if err := verifyModelSnapshot(snapshot, false); err != nil {
			entry.Status, entry.SourceError = "unavailable", err.Error()
			return entry
		}
		describeModelDirectory(&entry, snapshot.Path, false)
	}
	return entry
}

func huggingFaceHubCache() string {
	if cache := os.Getenv("HF_HUB_CACHE"); cache != "" {
		return cache
	}
	if home := huggingFaceHome(); home != "" {
		return filepath.Join(home, "hub")
	}
	return ""
}

// huggingFaceCacheEntries lists each cached revision (models--org--name/snapshots/<commit>).
func huggingFaceCacheEntries() []modelInventoryEntry {
	cache := huggingFaceHubCache()
	repos, _ := os.ReadDir(cache)
	entries := []modelInventoryEntry{}
	for _, repo := range repos {
		if !repo.IsDir() || !strings.HasPrefix(repo.Name(), "models--") {
			continue
		}
		name := strings.Replace(strings.TrimPrefix(repo.Name(), "models--"), "--", "/", 1)
		refs := map[string]string{}
		refDir := filepath.Join(cache, repo.Name(), "refs")
		if refNames, err := os.ReadDir(refDir); err == nil {
			for _, ref := range refNames {
				if data, err := os.ReadFile(filepath.Join(refDir, ref.Name())); err == nil {
					refs[strings.TrimSpace(string(data))] = ref.Name()
				}
			}
		}
		revisions, _ := os.ReadDir(filepath.Join(cache, repo.Name(), "snapshots"))
		for _, revision := range revisions {
			if !revision.IsDir() {
				continue
			}
			base := modelInventoryEntry{
				Kind: "hf-cache", Source: "hf-cache", Name: name, Repo: name, Revision: revision.Name(),
				Path: filepath.Join(cache, repo.Name(), "snapshots", revision.Name()),
			}
			if ref := refs[revision.Name()]; ref != "" {
				base.Source = "hf-cache (" + ref + ")"
			}
			for _, entry := range directoryModelEntries(base) {
				entry.ID = "hf:" + name + "@" + shortRevision(revision.Name())
				if entry.File != "" {
					entry.ID += ":" + entry.File
				}
				entries = append(entries, entry)
			}
		}
	}
	return entries
}

func modelScanDirs(args cliArgs) []string {
	var dirs []string
	for _, value := range []string{opt(args, "scan-dir"), os.Getenv("LMX_MODEL_DIRS")} {
		for _, part := range strings.FieldsFunc(value, func(r rune) bool { return r == ',' || r == os.PathListSeparator }) {
			if part = strings.TrimSpace(part); part != "" {
				if abs, err := filepath.Abs(expandHome(part)); err == nil {
					dirs = append(dirs, abs)
				}
			}
		}
	}
	return dirs
}

func expandHome(path string) string {
	if path == "~" || strings.HasPrefix(path, "~/") {
		if home, err := os.UserHomeDir(); err == nil {
			return filepath.Join(home, strings.TrimPrefix(path, "~"))
		}
	}
	return path
}

// scanDirEntries treats the directory itself and each immediate subdirectory
// as a possible model directory (Hugging Face local_dir downloads, GGUF folders).
func scanDirEntries(dir string) []modelInventoryEntry {
	entries := []modelInventoryEntry{}
	candidates := []string{dir}
	if children, err := os.ReadDir(dir); err == nil {
		for _, child := range children {
			if child.IsDir() && !strings.HasPrefix(child.Name(), ".") {
				candidates = append(candidates, filepath.Join(dir, child.Name()))
			}
		}
	} else {
		return []modelInventoryEntry{{ID: "dir:" + dir, Kind: "directory", Source: "directory", Name: filepath.Base(dir), Path: dir, Status: "unreadable", SourceError: err.Error()}}
	}
	for _, candidate := range candidates {
		if candidate == dir && !dirHasModelFiles(dir) {
			continue // the scan root is only a container of model directories
		}
		base := modelInventoryEntry{Kind: "directory", Source: "directory", Name: filepath.Base(candidate), Path: candidate, Revision: localDirRevision(candidate)}
		found := directoryModelEntries(base)
		for _, entry := range found {
			entry.ID = "dir:" + entry.Path
			entries = append(entries, entry)
		}
	}
	return entries
}

func dirHasModelFiles(dir string) bool {
	files, _ := os.ReadDir(dir)
	for _, file := range files {
		ext := strings.ToLower(filepath.Ext(file.Name()))
		if ext == ".gguf" || weightExtensions[ext] != "" || file.Name() == "config.json" {
			return true
		}
	}
	return false
}

// localDirRevision reads the commit huggingface_hub records for local_dir downloads.
func localDirRevision(dir string) string {
	matches, _ := filepath.Glob(filepath.Join(dir, ".cache", "huggingface", "download", "*.metadata"))
	for _, match := range matches {
		if data, err := os.ReadFile(match); err == nil {
			if line, _, _ := strings.Cut(string(data), "\n"); modelPullHex(strings.TrimSpace(line), 20) {
				return strings.TrimSpace(line)
			}
		}
	}
	return ""
}

// directoryModelEntries splits one directory into model entries: one for its
// safetensors/PyTorch weights, one per GGUF (split parts grouped), or a single
// no-weights entry.
func directoryModelEntries(base modelInventoryEntry) []modelInventoryEntry {
	files, err := os.ReadDir(base.Path)
	if err != nil {
		base.Status, base.SourceError = "unreadable", err.Error()
		return []modelInventoryEntry{base}
	}
	hasWeights := false
	ggufGroups := map[string][]string{}
	for _, file := range files {
		name := file.Name()
		if strings.HasSuffix(strings.ToLower(name), ".gguf") {
			group := name
			if match := ggufSplitPattern.FindStringSubmatchIndex(name); match != nil {
				group = name[:match[0]] + ".gguf"
			}
			ggufGroups[group] = append(ggufGroups[group], name)
			continue
		}
		if weightExtensions[strings.ToLower(filepath.Ext(name))] != "" {
			hasWeights = true
		}
	}
	entries := []modelInventoryEntry{}
	if hasWeights {
		entry := base
		describeModelDirectory(&entry, base.Path, true)
		entries = append(entries, entry)
	}
	groups := make([]string, 0, len(ggufGroups))
	for group := range ggufGroups {
		groups = append(groups, group)
	}
	sort.Strings(groups)
	for _, group := range groups {
		parts := ggufGroups[group]
		sort.Strings(parts)
		entry := base
		entry.File = parts[0]
		entry.Path = filepath.Join(base.Path, parts[0])
		entry.Name = base.Name + "/" + strings.TrimSuffix(group, ".gguf")
		entry.Status = "available"
		for _, part := range parts {
			info, err := os.Stat(filepath.Join(base.Path, part))
			if err != nil {
				entry.Status, entry.SourceError = "partial", err.Error()
				continue
			}
			entry.Size += info.Size()
			entry.FileCount++
			entry.Files = append(entry.Files, modelInventoryFile{Name: part, Size: info.Size()})
		}
		if match := ggufSplitPattern.FindStringSubmatch(parts[0]); match != nil && fmt.Sprintf("%05d", len(parts)) != match[2] {
			entry.Status, entry.SourceError = "partial", fmt.Sprintf("%d of %s split parts present", len(parts), strings.TrimLeft(match[2], "0"))
		}
		describeWeightFile(&entry, entry.Path)
		for _, part := range parts[1:] {
			if _, params, err := readGGUFSummary(filepath.Join(base.Path, part)); err == nil && entry.Parameters > 0 {
				entry.Parameters += params
			}
		}
		entries = append(entries, entry)
	}
	if len(entries) == 0 {
		base.Status = "no-weights"
		describeModelDirectory(&base, base.Path, false)
		entries = append(entries, base)
	}
	return entries
}

// describeModelDirectory fills size, files, format, config details and, for
// unquantized safetensors, the exact parameter count. weights marks a
// directory holding non-GGUF weight files.
func describeModelDirectory(entry *modelInventoryEntry, dir string, weights bool) {
	if entry.Status == "" {
		entry.Status = "available"
	}
	formats := map[string]bool{}
	var safetensors []string
	entry.Size, entry.FileCount, entry.Files = 0, 0, nil
	_ = filepath.WalkDir(dir, func(path string, d os.DirEntry, err error) error {
		if err != nil {
			return nil
		}
		rel, _ := filepath.Rel(dir, path)
		if d.IsDir() {
			if rel != "." && strings.HasPrefix(d.Name(), ".") {
				return filepath.SkipDir
			}
			return nil
		}
		info, err := os.Stat(path) // follows HF cache and snapshot links
		if err != nil {
			entry.Status, entry.SourceError = "partial", "unresolved link: "+filepath.ToSlash(rel)
			return nil
		}
		if strings.HasSuffix(strings.ToLower(d.Name()), ".gguf") && weights {
			return nil // GGUF files are listed as their own entries
		}
		entry.Size += info.Size()
		entry.FileCount++
		entry.Files = append(entry.Files, modelInventoryFile{Name: filepath.ToSlash(rel), Size: info.Size()})
		if format := weightExtensions[strings.ToLower(filepath.Ext(d.Name()))]; format != "" {
			formats[format] = true
			if format == "safetensors" {
				safetensors = append(safetensors, path)
			}
		}
		return nil
	})
	switch len(formats) {
	case 0:
	case 1:
		for format := range formats {
			entry.Format = format
		}
	default:
		entry.Format = "mixed"
	}
	config := readJSONObjectFile(filepath.Join(dir, "config.json"))
	if config != nil {
		applyConfigDetails(entry, config)
	}
	if index := readJSONObjectFile(filepath.Join(dir, "model.safetensors.index.json")); index != nil {
		missing := 0
		seen := map[string]bool{}
		for _, shard := range asObject(index["weight_map"]) {
			name := stringValue(shard)
			if name == "" || seen[name] {
				continue
			}
			seen[name] = true
			if _, err := os.Stat(filepath.Join(dir, filepath.FromSlash(name))); err != nil {
				missing++
			}
		}
		if missing > 0 {
			entry.Status, entry.SourceError = "partial", fmt.Sprintf("%d of %d safetensors shards missing", missing, len(seen))
		}
	}
	// Packed quantized tensors (AWQ/GPTQ/NVFP4) do not count parameters 1:1.
	if entry.Format == "safetensors" && (config == nil || config["quantization_config"] == nil) {
		if total, ok := safetensorsParameterCount(safetensors); ok {
			entry.Parameters = total
		}
	}
}

func readJSONObjectFile(path string) map[string]any {
	data, err := os.ReadFile(path)
	if err != nil || len(data) > 16<<20 {
		return nil
	}
	var value map[string]any
	if json.Unmarshal(data, &value) != nil {
		return nil
	}
	return value
}

// applyConfigDetails reads Hugging Face config.json fields, falling back to the
// nested text_config of multimodal models.
func applyConfigDetails(entry *modelInventoryEntry, config map[string]any) {
	text := asObject(config["text_config"])
	number := func(keys ...string) int64 {
		for _, source := range []map[string]any{config, text} {
			for _, key := range keys {
				if value, ok := source[key].(float64); ok && value > 0 {
					return int64(value)
				}
			}
		}
		return 0
	}
	if architectures := anySlice(config["architectures"]); len(architectures) > 0 {
		entry.Architecture = stringValue(architectures[0])
	}
	entry.ModelType = stringValue(config["model_type"])
	entry.ContextLength = number("max_position_embeddings", "max_sequence_length", "seq_length", "n_positions")
	entry.Layers = number("num_hidden_layers", "n_layer", "num_layers")
	entry.Experts = number("num_experts", "num_local_experts", "n_routed_experts")
	entry.ExpertsPerToken = number("num_experts_per_tok", "num_experts_per_token", "moe_topk")
	if quant, _, present := configQuantMethod(config); present {
		if quant != "" {
			entry.Quantization, entry.QuantizationSource = quant, "config quantization_config"
		} else {
			entry.Quantization, entry.QuantizationSource = quantizationSchemeLabel(asObject(config["quantization_config"])), "config quantization_config"
		}
	} else if quant := configDtype(config); quant != "" {
		entry.Quantization, entry.QuantizationSource = quant, "config dtype"
	}
}

// quantizationSchemeLabel names multi-format quantization configs by their
// scheme: compressed-tensors weight/activation bits (W4A16, W8A8, FP8, NVFP4) or
// ModelOpt's algorithm; otherwise the method name.
func quantizationSchemeLabel(qc map[string]any) string {
	method := strings.ToLower(stringValue(qc["quant_method"]))
	if method == "compressed-tensors" {
		for _, raw := range asObject(qc["config_groups"]) {
			group := asObject(raw)
			weights, activations := asObject(group["weights"]), asObject(group["input_activations"])
			bits, kind := int(numberField(weights, "num_bits")), stringValue(weights["type"])
			switch {
			case kind == "float" && bits == 8:
				return "FP8"
			case kind == "float" && bits == 4:
				return "NVFP4"
			case bits > 0 && activations == nil:
				return fmt.Sprintf("W%dA16", bits)
			case bits > 0:
				return fmt.Sprintf("W%dA%d", bits, int(numberField(activations, "num_bits")))
			}
		}
	}
	if method == "modelopt" {
		if algo := stringValue(qc["quant_algo"]); algo != "" {
			return strings.ReplaceAll(algo, "_", "-") + " (modelopt)"
		}
	}
	return strings.ToUpper(firstNonEmpty(method, "quantized"))
}

// safetensorsParameterCount sums tensor element counts from shard headers.
func safetensorsParameterCount(paths []string) (int64, bool) {
	if len(paths) == 0 {
		return 0, false
	}
	var total int64
	for _, path := range paths {
		file, err := os.Open(path)
		if err != nil {
			return 0, false
		}
		var length uint64
		err = binary.Read(file, binary.LittleEndian, &length)
		if err != nil || length > 100<<20 {
			file.Close()
			return 0, false
		}
		header := make([]byte, length)
		_, err = io.ReadFull(file, header)
		file.Close()
		if err != nil {
			return 0, false
		}
		var tensors map[string]json.RawMessage
		if json.Unmarshal(header, &tensors) != nil {
			return 0, false
		}
		for name, raw := range tensors {
			if name == "__metadata__" {
				continue
			}
			var tensor struct {
				Shape []int64 `json:"shape"`
			}
			if json.Unmarshal(raw, &tensor) != nil {
				return 0, false
			}
			count := int64(1)
			for _, dim := range tensor.Shape {
				count *= dim
			}
			total += count
		}
	}
	return total, total > 0
}

// describeWeightFile adds GGUF metadata (architecture, context, quantization,
// parameter count) or a filename-derived quantization for other files.
func describeWeightFile(entry *modelInventoryEntry, path string) {
	if !strings.HasSuffix(strings.ToLower(path), ".gguf") {
		if quant := quantizationFromFilename(path); quant != "" {
			entry.Quantization, entry.QuantizationSource = quant, "filename"
		}
		if format := weightExtensions[strings.ToLower(filepath.Ext(path))]; format != "" {
			entry.Format = format
		}
		return
	}
	entry.Format = "gguf"
	meta, params, err := readGGUFSummary(path)
	if err != nil {
		if entry.SourceError == "" {
			entry.SourceError = "GGUF metadata: " + err.Error()
		}
	} else {
		arch := stringValue(meta["general.architecture"])
		entry.Architecture = arch
		entry.SizeLabel = stringValue(meta["general.size_label"])
		entry.ContextLength = ggufInt(meta[arch+".context_length"])
		entry.Layers = ggufInt(meta[arch+".block_count"])
		entry.Experts = ggufInt(meta[arch+".expert_count"])
		entry.ExpertsPerToken = ggufInt(meta[arch+".expert_used_count"])
		entry.Parameters = params
		if fileType, ok := meta["general.file_type"].(uint32); ok && ggufFileTypeNames[fileType] != "" {
			entry.Quantization, entry.QuantizationSource = ggufFileTypeNames[fileType], "gguf file_type"
		}
	}
	if quant := quantizationFromFilename(path); quant != "" {
		entry.Quantization, entry.QuantizationSource = quant, "filename"
	}
}

func ggufInt(value any) int64 {
	switch v := value.(type) {
	case uint32:
		return int64(v)
	case int32:
		return int64(v)
	case uint64:
		if v <= math.MaxInt64 {
			return int64(v)
		}
	case int64:
		return v
	}
	return 0
}

// readGGUFSummary reads scalar metadata and sums tensor element counts from a
// GGUF header (v2/v3). Arrays such as the tokenizer vocabulary are skipped.
func readGGUFSummary(path string) (map[string]any, int64, error) {
	file, err := os.Open(path)
	if err != nil {
		return nil, 0, err
	}
	defer file.Close()
	r := bufio.NewReaderSize(io.LimitReader(file, 512<<20), 1<<20)
	var header struct {
		Magic   [4]byte
		Version uint32
		Tensors uint64
		KVs     uint64
	}
	if err := binary.Read(r, binary.LittleEndian, &header); err != nil {
		return nil, 0, err
	}
	if string(header.Magic[:]) != "GGUF" || header.Version < 2 || header.Tensors > 1<<24 || header.KVs > 1<<20 {
		return nil, 0, errors.New("not a supported GGUF file")
	}
	readString := func() (string, error) {
		var n uint64
		if err := binary.Read(r, binary.LittleEndian, &n); err != nil {
			return "", err
		}
		if n > 1<<24 {
			return "", errors.New("GGUF string too long")
		}
		buf := make([]byte, n)
		_, err := io.ReadFull(r, buf)
		return string(buf), err
	}
	scalarSize := map[uint32]int{0: 1, 1: 1, 2: 2, 3: 2, 4: 4, 5: 4, 6: 4, 7: 1, 10: 8, 11: 8, 12: 8}
	var readValue func(kind uint32, keep bool) (any, error)
	readValue = func(kind uint32, keep bool) (any, error) {
		switch kind {
		case 8:
			return readString()
		case 9:
			var itemKind uint32
			var count uint64
			if err := binary.Read(r, binary.LittleEndian, &itemKind); err != nil {
				return nil, err
			}
			if err := binary.Read(r, binary.LittleEndian, &count); err != nil {
				return nil, err
			}
			if count > 1<<28 {
				return nil, errors.New("GGUF array too long")
			}
			if size, ok := scalarSize[itemKind]; ok {
				_, err := r.Discard(int(count) * size)
				return nil, err
			}
			for i := uint64(0); i < count; i++ {
				if _, err := readValue(itemKind, false); err != nil {
					return nil, err
				}
			}
			return nil, nil
		}
		size, ok := scalarSize[kind]
		if !ok {
			return nil, fmt.Errorf("unknown GGUF value type %d", kind)
		}
		buf := make([]byte, size)
		if _, err := io.ReadFull(r, buf); err != nil {
			return nil, err
		}
		if !keep {
			return nil, nil
		}
		switch kind {
		case 4:
			return binary.LittleEndian.Uint32(buf), nil
		case 5:
			return int32(binary.LittleEndian.Uint32(buf)), nil
		case 10:
			return binary.LittleEndian.Uint64(buf), nil
		case 11:
			return int64(binary.LittleEndian.Uint64(buf)), nil
		}
		return nil, nil
	}
	meta := map[string]any{}
	for i := uint64(0); i < header.KVs; i++ {
		key, err := readString()
		if err != nil {
			return nil, 0, err
		}
		var kind uint32
		if err := binary.Read(r, binary.LittleEndian, &kind); err != nil {
			return nil, 0, err
		}
		value, err := readValue(kind, true)
		if err != nil {
			return nil, 0, err
		}
		if value != nil {
			meta[key] = value
		}
	}
	var params int64
	for i := uint64(0); i < header.Tensors; i++ {
		if _, err := readString(); err != nil {
			return meta, 0, nil // metadata is still useful without the count
		}
		var dims uint32
		if err := binary.Read(r, binary.LittleEndian, &dims); err != nil || dims > 8 {
			return meta, 0, nil
		}
		count := int64(1)
		for d := uint32(0); d < dims; d++ {
			var dim uint64
			if err := binary.Read(r, binary.LittleEndian, &dim); err != nil {
				return meta, 0, nil
			}
			count *= int64(dim)
		}
		if _, err := r.Discard(12); err != nil { // tensor type (u32) + data offset (u64)
			return meta, 0, nil
		}
		params += count
	}
	return meta, params, nil
}

// markModelUsage records which lmx model load servers serve each entry.
func markModelUsage(entries []modelInventoryEntry) {
	loaded, err := readLoadedModels()
	if err != nil {
		return
	}
	for _, m := range loaded {
		if !m.alive() {
			continue
		}
		for i := range entries {
			if entries[i].Path == m.ModelPath {
				entries[i].LoadedAt = append(entries[i].LoadedAt, m.BaseURL)
			}
		}
	}
}

func shortRevision(revision string) string {
	if len(revision) > 12 {
		return revision[:12]
	}
	return revision
}

func formatModelBytes(size int64) string {
	switch {
	case size >= 1<<30:
		return fmt.Sprintf("%.1f GiB", float64(size)/(1<<30))
	case size >= 1<<20:
		return fmt.Sprintf("%.0f MiB", float64(size)/(1<<20))
	case size >= 1<<10:
		return fmt.Sprintf("%.0f KiB", float64(size)/(1<<10))
	}
	return fmt.Sprintf("%d B", size)
}

func formatModelParameters(entry modelInventoryEntry) string {
	if entry.Parameters > 0 {
		value := float64(entry.Parameters)
		if value >= 1e9 {
			return fmt.Sprintf("%.1fB", value/1e9)
		}
		return fmt.Sprintf("%.0fM", value/1e6)
	}
	return entry.SizeLabel
}

func dash(value string) string {
	if value == "" {
		return "-"
	}
	return value
}

func handleModelList(args cliArgs) error {
	root, entries, err := modelInventory(args)
	if err != nil {
		return err
	}
	if !hasFlag(args, "all") {
		visible := entries[:0]
		for _, entry := range entries {
			if entry.Status != "no-weights" {
				visible = append(visible, entry)
			}
		}
		entries = visible
	}
	if kind := opt(args, "kind"); kind != "" {
		filtered := []modelInventoryEntry{}
		for _, entry := range entries {
			if entry.Kind == kind {
				filtered = append(filtered, entry)
			}
		}
		entries = filtered
	}
	for i := range entries {
		entries[i].Files = nil // per-file listings are shown by lmx model show
	}
	if !humanOutput(args) || opt(args, "out") != "" {
		return writeOrPrintJSON("model_list", args, map[string]any{"root": root, "models": entries})
	}
	if len(entries) == 0 {
		fmt.Println("No local models found. Pull one with lmx model pull/snapshot, register a file with lmx model import, or add --scan-dir <dir>.")
		return nil
	}
	nameCount := map[string]int{}
	for _, entry := range entries {
		nameCount[strings.ToLower(entry.Name)]++
	}
	writer := tabwriter.NewWriter(os.Stdout, 0, 4, 2, ' ', 0)
	fmt.Fprintln(writer, "NAME\tFORMAT\tQUANT\tPARAMS\tSIZE\tCONTEXT\tSTATUS\tSOURCE\tID")
	for _, entry := range entries {
		context := "-"
		if entry.ContextLength > 0 {
			context = formatModelContext(entry.ContextLength)
		}
		status := entry.Status
		if len(entry.LoadedAt) > 0 {
			status = "loaded " + strings.Join(entry.LoadedAt, ",")
		}
		// lmx IDs are needed for verify/remove/serving; other entries are shown by
		// name, with the full ID only when the name alone is ambiguous.
		id := "-"
		if entry.Kind == "file" || entry.Kind == "snapshot" {
			id = entry.ID[:12]
		} else if nameCount[strings.ToLower(entry.Name)] > 1 {
			id = entry.ID
		}
		fmt.Fprintf(writer, "%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n", entry.Name, dash(entry.Format), dash(entry.Quantization),
			dash(formatModelParameters(entry)), formatModelBytes(entry.Size), context, status, entry.Source, id)
	}
	writer.Flush()
	fmt.Printf("\n%d models. Details: lmx model show <name or id>. JSON: --json.\n", len(entries))
	return nil
}

func formatModelContext(tokens int64) string {
	if tokens >= 1000 && tokens%1000 == 0 {
		return fmt.Sprintf("%dK", tokens/1000)
	}
	if tokens >= 1024 && tokens%1024 == 0 {
		return fmt.Sprintf("%dK", tokens/1024)
	}
	return fmt.Sprint(tokens)
}

// handleModelShow prints every detail of one entry, selected by ID, a unique ID
// prefix (6+ characters), name or path.
func handleModelShow(target string, args cliArgs) error {
	if target == "" {
		return cliError{"missing_model", "model show requires a name, ID or path.", []string{"Run lmx model list to see local models."}, nil}
	}
	_, entries, err := modelInventory(args)
	if err != nil {
		return err
	}
	matches := matchInventory(entries, target)
	if len(matches) == 0 {
		return cliError{"model_not_found", "No local model matches " + target + ".", []string{"Run lmx model list (add --scan-dir <dir> for plain model folders)."}, nil}
	}
	if len(matches) > 1 {
		ids := []string{}
		for _, match := range matches {
			ids = append(ids, match.ID)
		}
		return cliError{"ambiguous_model", target + " matches several local models; pass an ID.", ids, nil}
	}
	entry := matches[0]
	if !humanOutput(args) || opt(args, "out") != "" {
		return writeOrPrintJSON("model_show", args, entry)
	}
	writer := tabwriter.NewWriter(os.Stdout, 0, 4, 2, ' ', 0)
	row := func(label string, value any) {
		if text := fmt.Sprint(value); text != "" && text != "0" && text != "false" {
			fmt.Fprintf(writer, "%s\t%s\n", label, text)
		}
	}
	row("Name", entry.Name)
	row("ID", entry.ID)
	row("Kind", entry.Kind)
	row("Source", entry.Source)
	row("Status", entry.Status+map[bool]string{true: " — " + entry.SourceError, false: ""}[entry.SourceError != ""])
	row("Repository", entry.Repo)
	row("Revision", entry.Revision)
	row("Path", entry.Path)
	row("Format", entry.Format)
	if entry.Quantization != "" {
		row("Quantization", entry.Quantization+" ("+entry.QuantizationSource+")")
	}
	row("Architecture", entry.Architecture)
	row("Model type", entry.ModelType)
	if entry.Parameters > 0 {
		row("Parameters", fmt.Sprintf("%s (%d)", formatModelParameters(entry), entry.Parameters))
	}
	row("Size label", entry.SizeLabel)
	if entry.ContextLength > 0 {
		row("Context length", fmt.Sprintf("%d tokens", entry.ContextLength))
	}
	row("Layers", entry.Layers)
	if entry.Experts > 0 {
		row("Experts", fmt.Sprintf("%d (%d per token)", entry.Experts, entry.ExpertsPerToken))
	}
	row("Size", fmt.Sprintf("%s (%d bytes, %d %s)", formatModelBytes(entry.Size), entry.Size, entry.FileCount, map[bool]string{true: "file", false: "files"}[entry.FileCount == 1]))
	row("SHA-256", entry.SHA256)
	if entry.Kind == "file" {
		row("Upstream verified", fmt.Sprint(entry.UpstreamVerified))
	}
	row("Loaded at", strings.Join(entry.LoadedAt, ", "))
	writer.Flush()
	if len(entry.Files) > 1 {
		fmt.Println("\nFiles:")
		files := tabwriter.NewWriter(os.Stdout, 0, 4, 2, ' ', tabwriter.AlignRight)
		for _, file := range entry.Files {
			fmt.Fprintf(files, "  %s\t  %s\n", formatModelBytes(file.Size), file.Name)
		}
		files.Flush()
	}
	return nil
}
