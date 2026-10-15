package main

import (
	"bytes"
	"encoding/binary"
	"encoding/json"
	"os"
	"path/filepath"
	"strings"
	"testing"
)

// writeTestGGUF writes a minimal GGUF v3 file: scalar metadata, a string array
// (as tokenizers have) and tensors whose element counts sum to the parameters.
func writeTestGGUF(t *testing.T, path string, tensors [][]uint64) {
	t.Helper()
	var b bytes.Buffer
	le := func(v any) { binary.Write(&b, binary.LittleEndian, v) }
	str := func(s string) { le(uint64(len(s))); b.WriteString(s) }
	b.WriteString("GGUF")
	le(uint32(3))
	le(uint64(len(tensors)))
	le(uint64(6))
	str("general.architecture")
	le(uint32(8))
	str("llama")
	str("llama.context_length")
	le(uint32(4))
	le(uint32(4096))
	str("llama.block_count")
	le(uint32(4))
	le(uint32(2))
	str("general.file_type")
	le(uint32(4))
	le(uint32(15))
	str("general.size_label")
	le(uint32(8))
	str("1B")
	str("tokenizer.ggml.tokens")
	le(uint32(9))
	le(uint32(8))
	le(uint64(3))
	for _, token := range []string{"<s>", "a", "b"} {
		str(token)
	}
	for i, dims := range tensors {
		str("t" + string(rune('0'+i)))
		le(uint32(len(dims)))
		for _, dim := range dims {
			le(dim)
		}
		le(uint32(0))
		le(uint64(0))
	}
	if err := os.WriteFile(path, b.Bytes(), 0o600); err != nil {
		t.Fatal(err)
	}
}

func writeTestSafetensors(t *testing.T, path string, shapes map[string][]int64) {
	t.Helper()
	header := map[string]any{"__metadata__": map[string]string{"format": "pt"}}
	for name, shape := range shapes {
		header[name] = map[string]any{"dtype": "BF16", "shape": shape, "data_offsets": []int{0, 0}}
	}
	data, _ := json.Marshal(header)
	var b bytes.Buffer
	binary.Write(&b, binary.LittleEndian, uint64(len(data)))
	b.Write(data)
	if err := os.WriteFile(path, b.Bytes(), 0o600); err != nil {
		t.Fatal(err)
	}
}

func inventoryByName(t *testing.T, args cliArgs) map[string]modelInventoryEntry {
	t.Helper()
	_, entries, err := modelInventory(args)
	if err != nil {
		t.Fatal(err)
	}
	byName := map[string]modelInventoryEntry{}
	for _, entry := range entries {
		byName[entry.Name] = entry
	}
	return byName
}

// Every source is listed with details read from the files themselves, and
// incomplete downloads are flagged instead of looking usable.
func TestModelInventoryDetailsAcrossSources(t *testing.T) {
	args := modelStoreTestArgs(t)
	scan := t.TempDir()
	args.opts["scan-dir"] = scan

	// Plain directory: safetensors + config, parameters from tensor shapes.
	dense := filepath.Join(scan, "dense")
	writeTestJSON(t, filepath.Join(dense, "config.json"), map[string]any{"architectures": []string{"LlamaForCausalLM"}, "model_type": "llama", "torch_dtype": "bfloat16", "max_position_embeddings": 32768, "num_hidden_layers": 4})
	writeTestSafetensors(t, filepath.Join(dense, "model.safetensors"), map[string][]int64{"a": {10, 10}, "b": {5}})

	// Sharded directory whose index names a shard that is not on disk.
	sharded := filepath.Join(scan, "sharded")
	writeTestJSON(t, filepath.Join(sharded, "config.json"), map[string]any{"torch_dtype": "float16"})
	writeTestSafetensors(t, filepath.Join(sharded, "model-00001-of-00002.safetensors"), map[string][]int64{"a": {2}})
	writeTestJSON(t, filepath.Join(sharded, "model.safetensors.index.json"), map[string]any{"weight_map": map[string]string{"a": "model-00001-of-00002.safetensors", "b": "model-00002-of-00002.safetensors"}})

	// GGUF folder: one complete quant, one split with a missing part.
	ggufDir := filepath.Join(scan, "gguf")
	os.MkdirAll(ggufDir, 0o755)
	writeTestGGUF(t, filepath.Join(ggufDir, "tiny.gguf"), [][]uint64{{4, 8}, {8}})
	writeTestGGUF(t, filepath.Join(ggufDir, "big-Q8_0-00001-of-00002.gguf"), [][]uint64{{2}})

	// Folder without weights is hidden from the default listing.
	writeTestJSON(t, filepath.Join(scan, "metadata-only", "config.json"), map[string]any{})

	// Hugging Face cache revision linked into blobs, with a compressed-tensors config.
	hub := os.Getenv("HF_HUB_CACHE")
	revision := strings.Repeat("c", 40)
	repo := filepath.Join(hub, "models--org--quantized")
	writeTestJSON(t, filepath.Join(repo, "snapshots", revision, "config.json"), map[string]any{"quantization_config": map[string]any{"quant_method": "compressed-tensors", "config_groups": map[string]any{"group_0": map[string]any{"weights": map[string]any{"num_bits": 4, "type": "int"}}}}})
	os.MkdirAll(filepath.Join(repo, "blobs"), 0o755)
	writeTestSafetensors(t, filepath.Join(repo, "blobs", "weights"), map[string][]int64{"a": {1}})
	os.Symlink(filepath.Join(repo, "blobs", "weights"), filepath.Join(repo, "snapshots", revision, "model.safetensors"))
	os.MkdirAll(filepath.Join(repo, "refs"), 0o755)
	os.WriteFile(filepath.Join(repo, "refs", "main"), []byte(revision), 0o600)

	got := inventoryByName(t, args)
	if e := got["dense"]; e.Status != "available" || e.Format != "safetensors" || e.Quantization != "BF16" || e.Parameters != 105 || e.ContextLength != 32768 || e.Layers != 4 || e.Architecture != "LlamaForCausalLM" {
		t.Fatalf("dense directory = %+v", e)
	}
	if e := got["sharded"]; e.Status != "partial" || !strings.Contains(e.SourceError, "1 of 2 safetensors shards missing") {
		t.Fatalf("missing shard not flagged: %+v", e)
	}
	if e := got["gguf/tiny"]; e.Format != "gguf" || e.Quantization != "Q4_K_M" || e.QuantizationSource != "gguf file_type" || e.Architecture != "llama" || e.ContextLength != 4096 || e.Layers != 2 || e.Parameters != 40 || e.SizeLabel != "1B" {
		t.Fatalf("GGUF metadata = %+v", e)
	}
	if e := got["gguf/big-Q8_0"]; e.Status != "partial" || e.Quantization != "Q8_0" {
		t.Fatalf("incomplete split GGUF = %+v", e)
	}
	if e := got["org/quantized"]; e.Kind != "hf-cache" || e.Source != "hf-cache (main)" || e.Quantization != "W4A16" || e.Parameters != 0 || e.Revision != revision {
		t.Fatalf("HF cache entry = %+v", e)
	}
	if e, ok := got["metadata-only"]; !ok || e.Status != "no-weights" {
		t.Fatalf("metadata-only folder should be inventoried as no-weights: %+v", e)
	}

	// The default listing hides folders without weights; --all shows them.
	listed := func(all bool) []modelInventoryEntry {
		args.flags["all"] = all
		if err := handleModelList(args); err != nil {
			t.Fatal(err)
		}
		var out struct {
			Models []modelInventoryEntry `json:"models"`
		}
		data, _ := os.ReadFile(opt(args, "out"))
		json.Unmarshal(data, &out)
		return out.Models
	}
	for _, e := range listed(false) {
		if e.Status == "no-weights" {
			t.Fatal("no-weights folder listed by default")
		}
	}
	if len(listed(true)) != len(got) {
		t.Fatal("--all must list every inventory entry")
	}
}

// show resolves names and ID prefixes, and refuses to guess between duplicates.
func TestModelShowSelection(t *testing.T) {
	args := modelStoreTestArgs(t)
	first, second := t.TempDir(), t.TempDir()
	args.opts["scan-dir"] = first + "," + second
	for _, dir := range []string{first, second} {
		writeTestJSON(t, filepath.Join(dir, "same", "config.json"), map[string]any{"torch_dtype": "bfloat16"})
		writeTestSafetensors(t, filepath.Join(dir, "same", "model.safetensors"), map[string][]int64{"a": {1}})
	}
	source := filepath.Join(t.TempDir(), "imported-Q4_K_M.gguf")
	writeTestGGUF(t, source, [][]uint64{{3}})
	if err := handleModelImport(source, args); err != nil {
		t.Fatal(err)
	}
	if err := handleModelShow("same", args); err == nil || !strings.Contains(err.Error(), "several local models") {
		t.Fatalf("ambiguous name was resolved: %v", err)
	}
	if err := handleModelShow("dir:"+filepath.Join(first, "same"), args); err != nil {
		t.Fatalf("exact ID: %v", err)
	}
	id := modelArtifactID("", "", source)
	if err := handleModelShow(id[:8], args); err != nil {
		t.Fatalf("ID prefix: %v", err)
	}
	var shown modelInventoryEntry
	data, _ := os.ReadFile(opt(args, "out"))
	json.Unmarshal(data, &shown)
	if shown.ID != id || shown.Kind != "file" || shown.Source != "imported" || shown.Quantization != "Q4_K_M" || shown.Parameters != 3 || shown.SHA256 == "" {
		t.Fatalf("imported GGUF details = %+v", shown)
	}
}
