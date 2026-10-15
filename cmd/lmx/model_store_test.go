package main

import (
	"crypto/sha256"
	"encoding/hex"
	"encoding/json"
	"errors"
	"os"
	"path/filepath"
	"testing"
)

func modelStoreTestArgs(t *testing.T) cliArgs {
	t.Helper()
	dir := t.TempDir()
	// Model listing also reads the HF cache, scan dirs and deployments; keep them empty.
	t.Setenv("HF_HUB_CACHE", filepath.Join(dir, "hf-hub"))
	t.Setenv("LMX_MODEL_DIRS", "")
	t.Setenv("LMX_DEPLOYMENTS_DIR", filepath.Join(dir, "servers"))
	return cliArgs{opts: map[string]string{"models-dir": filepath.Join(dir, "models"), "out": filepath.Join(dir, "result.json")}, flags: map[string]bool{}}
}

func TestModelImportRemovalPreservesSymlinkSource(t *testing.T) {
	args := modelStoreTestArgs(t)
	source := filepath.Join(t.TempDir(), "model with spaces.gguf")
	original := []byte("external model bytes")
	if err := os.WriteFile(source, original, 0o600); err != nil {
		t.Fatal(err)
	}
	link := filepath.Join(t.TempDir(), "HF cache link.gguf")
	if err := os.Symlink(source, link); err != nil {
		t.Fatal(err)
	}
	for _, target := range []string{link, source, link} {
		if err := handleModelImport(target, args); err != nil {
			t.Fatal(err)
		}
	}
	if err := handleModelList(args); err != nil {
		t.Fatal(err)
	}
	var inventory struct {
		Models []modelArtifact `json:"models"`
	}
	data, err := os.ReadFile(opt(args, "out"))
	if err != nil {
		t.Fatal(err)
	}
	if err := json.Unmarshal(data, &inventory); err != nil {
		t.Fatal(err)
	}
	if len(inventory.Models) != 1 {
		t.Fatalf("repeated imports created %d records", len(inventory.Models))
	}
	artifact := inventory.Models[0]
	if artifact.Managed || artifact.Repo != "" || artifact.Revision != "" || artifact.UpstreamVerified {
		t.Fatalf("import claimed ownership or upstream provenance: %+v", artifact)
	}
	if err := handleModelRemove(artifact.ID, args); err == nil {
		t.Fatal("removal without --yes succeeded")
	}
	if err := handleModelVerify(artifact.ID, args); err != nil {
		t.Fatalf("unconfirmed removal affected registration: %v", err)
	}
	args.flags["yes"] = true
	if err := handleModelRemove(artifact.ID, args); err != nil {
		t.Fatal(err)
	}
	data, err = os.ReadFile(link)
	if err != nil || string(data) != string(original) {
		t.Fatalf("external source changed after removal: %q, %v", data, err)
	}
	if err := handleModelVerify(artifact.ID, args); !errors.Is(err, os.ErrNotExist) {
		t.Fatalf("registration survived removal: %v", err)
	}
}

func TestModelVerifyDetectsSameSizeMutation(t *testing.T) {
	args := modelStoreTestArgs(t)
	source := filepath.Join(t.TempDir(), "model.gguf")
	if err := os.WriteFile(source, []byte("original"), 0o600); err != nil {
		t.Fatal(err)
	}
	if err := handleModelImport(source, args); err != nil {
		t.Fatal(err)
	}
	id := modelArtifactID("", "", source)
	if err := handleModelVerify(id, args); err != nil {
		t.Fatal(err)
	}
	if err := os.WriteFile(source, []byte("modified"), 0o600); err != nil {
		t.Fatal(err)
	}
	var mismatch cliError
	if err := handleModelVerify(id, args); !errors.As(err, &mismatch) || mismatch.Code != "model_checksum_mismatch" {
		t.Fatalf("same-size mutation was not reported as checksum mismatch: %v", err)
	}
	if err := os.Remove(source); err != nil {
		t.Fatal(err)
	}
	if err := handleModelList(args); err != nil {
		t.Fatal(err)
	}
	var inventory struct {
		Models []struct {
			ID     string `json:"id"`
			Status string `json:"status"`
		} `json:"models"`
	}
	data, err := os.ReadFile(opt(args, "out"))
	if err != nil {
		t.Fatal(err)
	}
	if err := json.Unmarshal(data, &inventory); err != nil {
		t.Fatal(err)
	}
	if len(inventory.Models) != 1 || inventory.Models[0].ID != id || inventory.Models[0].Status != "missing" {
		t.Fatalf("missing source was not listed visibly: %+v", inventory)
	}
}

func TestModelRemoveRejectsForgedOwnedPaths(t *testing.T) {
	args := modelStoreTestArgs(t)
	args.flags["yes"] = true
	root := opt(args, "models-dir")
	external := filepath.Join(t.TempDir(), "keep.gguf")
	original := []byte("never delete external data")
	if err := os.WriteFile(external, original, 0o600); err != nil {
		t.Fatal(err)
	}
	digest := sha256.Sum256(original)
	artifact := modelArtifact{Repo: "owner/repo", Revision: "commit", File: "model.gguf", Path: external, SHA256: hex.EncodeToString(digest[:]), Size: int64(len(original)), Managed: true}
	artifact.ID = modelArtifactID(artifact.Repo, artifact.Revision, artifact.File)
	if err := saveModelArtifact(root, artifact); err == nil {
		t.Fatal("accepted a managed registration pointing outside owned storage")
	}
	registration, err := modelRegistrationPath(root, artifact.ID)
	if err != nil {
		t.Fatal(err)
	}
	if err := os.MkdirAll(filepath.Dir(registration), 0o700); err != nil {
		t.Fatal(err)
	}
	data, err := json.Marshal(artifact)
	if err != nil {
		t.Fatal(err)
	}
	if err := os.WriteFile(registration, data, 0o600); err != nil {
		t.Fatal(err)
	}
	if err := handleModelRemove(artifact.ID, args); err == nil {
		t.Fatal("removed a forged managed registration")
	}
	owned, err := modelArtifactPath(root, artifact.Repo, artifact.Revision, artifact.File)
	if err != nil {
		t.Fatal(err)
	}
	if err := os.MkdirAll(filepath.Dir(owned), 0o700); err != nil {
		t.Fatal(err)
	}
	if err := os.Symlink(external, owned); err != nil {
		t.Fatal(err)
	}
	artifact.Path = owned
	data, err = json.Marshal(artifact)
	if err != nil {
		t.Fatal(err)
	}
	if err := os.WriteFile(registration, data, 0o600); err != nil {
		t.Fatal(err)
	}
	if err := handleModelRemove(artifact.ID, args); err == nil {
		t.Fatal("accepted a symlink as managed data")
	}
	data, err = os.ReadFile(external)
	if err != nil || string(data) != string(original) {
		t.Fatalf("external bytes were not preserved: %q, %v", data, err)
	}
}

func TestModelImportDryRunDoesNotCreateStore(t *testing.T) {
	args := modelStoreTestArgs(t)
	args.flags["dry-run"] = true
	source := filepath.Join(t.TempDir(), "model.gguf")
	if err := os.WriteFile(source, []byte("model"), 0o600); err != nil {
		t.Fatal(err)
	}
	if err := handleModelImport(source, args); err != nil {
		t.Fatal(err)
	}
	if _, err := os.Stat(opt(args, "models-dir")); !errors.Is(err, os.ErrNotExist) {
		t.Fatalf("dry-run created model storage: %v", err)
	}
}

func TestModelRemoveDeletesOnlyOwnedFile(t *testing.T) {
	args := modelStoreTestArgs(t)
	args.flags["yes"] = true
	root := opt(args, "models-dir")
	digest := sha256.Sum256([]byte("model"))
	artifact := modelArtifact{Repo: "owner/repo", Revision: "commit", File: "nested/model.gguf", SHA256: hex.EncodeToString(digest[:]), Size: 5, Managed: true}
	artifact.ID = modelArtifactID(artifact.Repo, artifact.Revision, artifact.File)
	var err error
	artifact.Path, err = modelArtifactPath(root, artifact.Repo, artifact.Revision, artifact.File)
	if err != nil {
		t.Fatal(err)
	}
	if err := os.MkdirAll(filepath.Dir(artifact.Path), 0o700); err != nil {
		t.Fatal(err)
	}
	sibling := filepath.Join(filepath.Dir(artifact.Path), "unrelated.keep")
	for _, filename := range []string{artifact.Path, sibling} {
		if err := os.WriteFile(filename, []byte("model"), 0o600); err != nil {
			t.Fatal(err)
		}
	}
	if err := saveModelArtifact(root, artifact); err != nil {
		t.Fatal(err)
	}
	if err := handleModelRemove(artifact.ID, args); err != nil {
		t.Fatal(err)
	}
	if _, err := os.Stat(artifact.Path); !errors.Is(err, os.ErrNotExist) {
		t.Fatalf("managed file survived removal: %v", err)
	}
	if data, err := os.ReadFile(sibling); err != nil || string(data) != "model" {
		t.Fatalf("removal affected unrelated sibling: %q, %v", data, err)
	}
	// An upstream-identified cache artifact still belongs to the external cache.
	artifact.Managed = false
	artifact.Path = sibling
	if err := saveModelArtifact(root, artifact); err != nil {
		t.Fatal(err)
	}
	if err := handleModelRemove(artifact.ID, args); err != nil {
		t.Fatal(err)
	}
	if data, err := os.ReadFile(sibling); err != nil || string(data) != "model" {
		t.Fatalf("removal affected an externally referenced cache file: %q, %v", data, err)
	}
}
