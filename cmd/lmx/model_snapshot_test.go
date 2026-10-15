package main

import (
	"encoding/json"
	"fmt"
	"net/http"
	"net/http/httptest"
	"os"
	"path/filepath"
	"sort"
	"strings"
	"testing"
)

func modelSnapshotTestServer(t *testing.T, files map[string]string, download func(http.ResponseWriter, *http.Request, string, string)) *httptest.Server {
	t.Helper()
	commit := strings.Repeat("a", 40)
	return httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if r.URL.Path == "/api/models/owner/model/revision/main" {
			var siblings []any
			for name, data := range files {
				source := modelPullTestSource(data)
				siblings = append(siblings, map[string]any{"rfilename": name, "size": source.Size, "lfs": map[string]any{"sha256": source.ExpectedHash, "size": source.Size}})
			}
			json.NewEncoder(w).Encode(map[string]any{"sha": commit, "siblings": siblings})
			return
		}
		prefix := "/owner/model/resolve/" + commit + "/"
		if !strings.HasPrefix(r.URL.Path, prefix) {
			t.Errorf("unpinned snapshot request: %s", r.URL.Path)
			http.NotFound(w, r)
			return
		}
		name := strings.TrimPrefix(r.URL.Path, prefix)
		data, ok := files[name]
		if !ok {
			http.NotFound(w, r)
			return
		}
		download(w, r, name, data)
	}))
}

func modelSnapshotTestResult(t *testing.T, args cliArgs) modelSnapshot {
	t.Helper()
	data, err := os.ReadFile(opt(args, "out"))
	if err != nil {
		t.Fatal(err)
	}
	var result struct {
		Snapshot modelSnapshot `json:"snapshot"`
	}
	if err := json.Unmarshal(data, &result); err != nil {
		t.Fatal(err)
	}
	return result.Snapshot
}

func TestModelSnapshotInterruptedAcquisitionNeverLoadsAndResumes(t *testing.T) {
	files := map[string]string{"config.json": "{}", "weights/model.bin": "complete weights bytes"}
	requests := map[string]int{}
	server := modelSnapshotTestServer(t, files, func(w http.ResponseWriter, r *http.Request, name, data string) {
		requests[name]++
		if name == "weights/model.bin" && requests[name] == 1 {
			w.Header().Set("Content-Length", fmt.Sprint(len(data)))
			fmt.Fprint(w, data[:5])
			return
		}
		if name == "weights/model.bin" {
			if r.Header.Get("Range") != "bytes=5-" {
				t.Errorf("resume range = %q", r.Header.Get("Range"))
			}
			w.Header().Set("Content-Range", fmt.Sprintf("bytes 5-%d/%d", len(data)-1, len(data)))
			w.WriteHeader(http.StatusPartialContent)
			fmt.Fprint(w, data[5:])
			return
		}
		fmt.Fprint(w, data)
	})
	defer server.Close()
	root := filepath.Join(t.TempDir(), "store")
	args := modelPullTestArgs(t, server.URL, root)
	if err := handleModelSnapshot("owner/model", args); err == nil {
		t.Fatal("partial repository accepted")
	}
	snapshots, err := modelSnapshotInventory(root)
	if err != nil || len(snapshots) != 1 {
		t.Fatalf("partial inventory = %+v, %v", snapshots, err)
	}
	snapshot := snapshots[0]
	if _, err := loadModelSnapshot(args, snapshot.ID); err == nil {
		t.Fatal("partial snapshot loaded as ready")
	}
	if _, err := os.Stat(filepath.Join(root, "snapshots", snapshot.ID, "manifest.json")); !os.IsNotExist(err) {
		t.Fatalf("partial snapshot has ready manifest: %v", err)
	}
	if err := handleModelRemove(modelArtifactID("owner/model", snapshot.Revision, "config.json"), args); err == nil {
		t.Fatal("partial snapshot did not protect acquired config")
	}
	for range 2 {
		if err := handleModelSnapshot("owner/model", args); err != nil {
			t.Fatal(err)
		}
	}
	if requests["config.json"] != 1 || requests["weights/model.bin"] != 2 {
		t.Fatalf("repeat downloads: %v", requests)
	}
	ready, err := loadModelSnapshot(args, snapshot.ID)
	if err != nil {
		t.Fatal(err)
	}
	for name, expected := range files {
		filename := filepath.Join(ready.Path, filepath.FromSlash(name))
		if info, err := os.Lstat(filename); err != nil || info.Mode()&os.ModeSymlink == 0 {
			t.Fatalf("not a shared-file link: %s, %v", filename, err)
		}
		data, err := os.ReadFile(filename)
		if err != nil || string(data) != expected {
			t.Fatalf("serving file %s = %q, %v", name, data, err)
		}
	}
	server.Close()
	if err := handleModelSnapshots("verify", ready.ID, args); err != nil {
		t.Fatalf("offline verify: %v", err)
	}
	if err := handleModelSnapshots("list", "", args); err != nil {
		t.Fatalf("offline list: %v", err)
	}
	if err := os.Remove(ready.Files[0].Artifact.Path); err != nil {
		t.Fatal(err)
	}
	if _, err := loadModelSnapshot(args, ready.ID); err == nil {
		t.Fatal("ready record with a missing shared file loaded")
	}
}

func TestModelSnapshotHashFailureCannotBecomeReady(t *testing.T) {
	server := modelSnapshotTestServer(t, map[string]string{"model.bin": "correct weights"}, func(w http.ResponseWriter, _ *http.Request, _, data string) {
		fmt.Fprint(w, strings.Repeat("x", len(data)))
	})
	defer server.Close()
	root := filepath.Join(t.TempDir(), "store")
	args := modelPullTestArgs(t, server.URL, root)
	if err := handleModelSnapshot("owner/model", args); err == nil {
		t.Fatal("corrupt snapshot accepted")
	}
	snapshots, err := modelSnapshotInventory(root)
	if err != nil || len(snapshots) != 1 {
		t.Fatalf("inventory = %+v, %v", snapshots, err)
	}
	if _, err := loadModelSnapshot(args, snapshots[0].ID); err == nil {
		t.Fatal("corrupt snapshot loaded")
	}
	if err := handleModelSnapshots("remove", snapshots[0].ID, args); err != nil {
		t.Fatalf("remove incomplete: %v", err)
	}
}

func TestModelSnapshotPreviewSelectionAndInternalNames(t *testing.T) {
	files := map[string]string{"config.json": "{}", "manifest.json": "upstream", "pending.json": "upstream pending", "weights/fp16.bin": "float", "weights/q4.bin": "quant", "nested/tokenizer.json": "tokens"}
	server := modelSnapshotTestServer(t, files, func(w http.ResponseWriter, _ *http.Request, _, data string) { fmt.Fprint(w, data) })
	defer server.Close()
	root := filepath.Join(t.TempDir(), "store")
	args := modelPullTestArgs(t, server.URL, root)
	args.flags["yes"] = false
	if err := handleModelSnapshot("owner/model", args); err != nil {
		t.Fatal(err)
	}
	plan := modelSnapshotTestResult(t, args)
	names := make([]string, 0, len(files))
	var size int64
	for name, data := range files {
		names = append(names, name)
		size += int64(len(data))
	}
	sort.Strings(names)
	if len(plan.Files) != len(names) || plan.Size != size {
		t.Fatalf("incomplete default preview: %+v", plan)
	}
	for i, name := range names {
		if plan.Files[i].File != name || plan.Files[i].ExpectedHash == "" {
			t.Fatalf("missing exact name/hash for %s", name)
		}
	}
	if _, err := os.Stat(root); !os.IsNotExist(err) {
		t.Fatalf("preview mutated store: %v", err)
	}
	args.opts["include"] = "**/*.json,weights/*.bin"
	args.opts["exclude"] = "**/q4.bin"
	args.flags["yes"] = true
	if err := handleModelSnapshot("owner/model", args); err != nil {
		t.Fatal(err)
	}
	snapshot := modelSnapshotTestResult(t, args)
	if _, err := os.Stat(filepath.Join(snapshot.Path, "weights", "q4.bin")); !os.IsNotExist(err) {
		t.Fatalf("excluded variant exists: %v", err)
	}
	for _, name := range []string{"manifest.json", "pending.json", "nested/tokenizer.json", "weights/fp16.bin"} {
		data, err := os.ReadFile(filepath.Join(snapshot.Path, filepath.FromSlash(name)))
		if err != nil || string(data) != files[name] {
			t.Fatalf("upstream name collided: %s %q %v", name, data, err)
		}
	}
}

func TestModelSnapshotRejectsTraversalAndFileDirectoryCollisions(t *testing.T) {
	for _, files := range []map[string]string{{"../escape": "bad"}, {"/absolute": "bad"}, {"weights": "file", "weights/a.bin": "nested"}, {"bad\\name": "bad"}} {
		server := modelSnapshotTestServer(t, files, func(http.ResponseWriter, *http.Request, string, string) { t.Error("unsafe names downloaded") })
		args := modelPullTestArgs(t, server.URL, filepath.Join(t.TempDir(), "store"))
		if err := handleModelSnapshot("owner/model", args); err == nil {
			t.Fatalf("accepted unsafe files %v", files)
		}
		server.Close()
	}
	for _, glob := range []string{"../*", "[", "a,,b", "/absolute"} {
		if _, err := modelSnapshotPatterns(glob); err == nil {
			t.Fatalf("accepted unsafe glob %q", glob)
		}
	}
}

func TestModelSnapshotRemovalPreservesSharedArtifactsAndCache(t *testing.T) {
	files := map[string]string{"config.json": "{}", "model.bin": "weights"}
	downloads := 0
	server := modelSnapshotTestServer(t, files, func(w http.ResponseWriter, _ *http.Request, _, data string) { downloads++; fmt.Fprint(w, data) })
	defer server.Close()
	root := filepath.Join(t.TempDir(), "store")
	args := modelPullTestArgs(t, server.URL, root)
	cache := t.TempDir()
	t.Setenv("HF_HUB_CACHE", cache)
	cachedSource := modelPullTestSource(files["model.bin"])
	cached := filepath.Join(cache, "models--owner--model", "blobs", cachedSource.ExpectedHash)
	if err := os.MkdirAll(filepath.Dir(cached), 0o755); err != nil {
		t.Fatal(err)
	}
	if err := os.WriteFile(cached, []byte(files["model.bin"]), 0o600); err != nil {
		t.Fatal(err)
	}
	if err := handleModelSnapshot("owner/model", args); err != nil {
		t.Fatal(err)
	}
	snapshot := modelSnapshotTestResult(t, args)
	if downloads != 1 {
		t.Fatalf("cache file was downloaded: %d downloads", downloads)
	}
	if err := handleModelRemove(snapshot.Files[0].Artifact.ID, args); err == nil {
		t.Fatal("removed referenced artifact")
	}
	if err := handleModelSnapshots("remove", snapshot.ID, args); err != nil {
		t.Fatal(err)
	}
	for _, file := range snapshot.Files {
		data, err := os.ReadFile(file.Artifact.Path)
		if err != nil || string(data) != files[file.File] {
			t.Fatalf("removal damaged shared file %s: %v", file.File, err)
		}
		if _, err := readModelArtifact(root, file.Artifact.ID); err != nil {
			t.Fatalf("removal unregistered shared artifact: %v", err)
		}
	}
	if _, err := loadModelSnapshot(args, snapshot.ID); !os.IsNotExist(err) {
		t.Fatalf("removed snapshot still loads: %v", err)
	}
	if err := handleModelRemove(snapshot.Files[0].Artifact.ID, args); err != nil {
		t.Fatalf("removed snapshot left artifact reference: %v", err)
	}
}

func TestModelSnapshotRefusesUnownedLinksAndPreservesUnrelatedFiles(t *testing.T) {
	server := modelSnapshotTestServer(t, map[string]string{"config.json": "{}"}, func(w http.ResponseWriter, _ *http.Request, _, data string) { fmt.Fprint(w, data) })
	defer server.Close()
	args := modelPullTestArgs(t, server.URL, filepath.Join(t.TempDir(), "store"))
	if err := handleModelSnapshot("owner/model", args); err != nil {
		t.Fatal(err)
	}
	snapshot := modelSnapshotTestResult(t, args)
	unrelated := filepath.Join(snapshot.Path, "unrelated")
	if err := os.WriteFile(unrelated, []byte("keep"), 0o600); err != nil {
		t.Fatal(err)
	}
	if err := handleModelSnapshots("remove", snapshot.ID, args); err == nil {
		t.Fatal("snapshot removal accepted unrelated file")
	}
	if data, err := os.ReadFile(unrelated); err != nil || string(data) != "keep" {
		t.Fatalf("unrelated file damaged: %q %v", data, err)
	}
	if err := os.Remove(unrelated); err != nil {
		t.Fatal(err)
	}
	link := filepath.Join(snapshot.Path, "config.json")
	if err := os.Remove(link); err != nil {
		t.Fatal(err)
	}
	if err := os.Symlink("/etc/hosts", link); err != nil {
		t.Fatal(err)
	}
	if _, err := loadModelSnapshot(args, snapshot.ID); err == nil {
		t.Fatal("foreign link loaded")
	}
	if err := handleModelSnapshots("remove", snapshot.ID, args); err == nil {
		t.Fatal("foreign link removal accepted")
	}
}

func TestModelSnapshotServingLockOnlyBlocksItsOwnRemoval(t *testing.T) {
	server := modelSnapshotTestServer(t, map[string]string{"config.json": "{}", "model.bin": "weights"}, func(w http.ResponseWriter, _ *http.Request, _, data string) { fmt.Fprint(w, data) })
	defer server.Close()
	args := modelPullTestArgs(t, server.URL, filepath.Join(t.TempDir(), "store"))
	args.opts["include"] = "config.json"
	if err := handleModelSnapshot("owner/model", args); err != nil {
		t.Fatal(err)
	}
	first := modelSnapshotTestResult(t, args)
	release, err := lockModelSnapshot(args, first.ID)
	if err != nil {
		t.Fatal(err)
	}
	defer func() {
		if release != nil {
			release()
		}
	}()
	args.opts["include"] = "model.bin"
	if err := handleModelSnapshot("owner/model", args); err != nil {
		t.Fatalf("serving blocked another snapshot: %v", err)
	}
	second := modelSnapshotTestResult(t, args)
	if err := handleModelSnapshots("remove", second.ID, args); err != nil {
		t.Fatalf("serving blocked unrelated removal: %v", err)
	}
	if err := handleModelSnapshots("remove", first.ID, args); err == nil {
		t.Fatal("removed in-use snapshot")
	}
	if _, err := loadModelSnapshot(args, first.ID); err != nil {
		t.Fatalf("failed removal damaged serving snapshot: %v", err)
	}
	release()
	release = nil
	if err := handleModelSnapshots("remove", first.ID, args); err != nil {
		t.Fatalf("released snapshot could not be removed: %v", err)
	}
}
