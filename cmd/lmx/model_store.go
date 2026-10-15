package main

import (
	"crypto/sha256"
	"encoding/hex"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"os"
	"path"
	"path/filepath"
	"strings"
)

type modelArtifact struct {
	ID               string `json:"id"`
	Repo             string `json:"repo"`
	Revision         string `json:"revision"`
	File             string `json:"file"`
	Path             string `json:"path"`
	SHA256           string `json:"sha256"`
	Size             int64  `json:"size"`
	Managed          bool   `json:"managed"`
	Integrity        string `json:"integrity,omitempty"`
	ExpectedHash     string `json:"expectedHash,omitempty"`
	UpstreamVerified bool   `json:"upstreamVerified,omitempty"`
}

func modelStoreRoot(args cliArgs) (string, error) {
	root := opt(args, "models-dir")
	if root == "" {
		root = os.Getenv("LMX_MODELS_DIR")
	}
	if root == "" {
		base := os.Getenv("XDG_DATA_HOME")
		if base == "" {
			home, err := os.UserHomeDir()
			if err != nil {
				return "", err
			}
			base = filepath.Join(home, ".local", "share")
		}
		root = filepath.Join(base, "localmaxxing", "models")
	}
	return filepath.Abs(root)
}

func modelArtifactID(repo, revision, file string) string {
	data, _ := json.Marshal([3]string{repo, revision, file})
	sum := sha256.Sum256(data)
	return hex.EncodeToString(sum[:])
}

// Model store descendants must never redirect writes or removals through symlinks.
// The caller-selected root's ancestors are not a security sandbox.
func modelStoreCheckedPath(root string, parts ...string) (string, error) {
	current, err := filepath.Abs(root)
	if err != nil {
		return "", err
	}
	paths := []string{current}
	for _, part := range parts {
		if part == "" || part == "." || part == ".." || strings.ContainsAny(part, `/\\`) {
			return "", fmt.Errorf("invalid model store path component %q", part)
		}
		current = filepath.Join(current, part)
		paths = append(paths, current)
	}
	for _, candidate := range paths {
		info, err := os.Lstat(candidate)
		if errors.Is(err, os.ErrNotExist) {
			continue
		}
		if err != nil {
			return "", err
		}
		if info.Mode()&os.ModeSymlink != 0 {
			return "", fmt.Errorf("model store path must not be a symlink: %s", candidate)
		}
	}
	return current, nil
}

func modelArtifactPath(root, repo, revision, file string) (string, error) {
	if repo == "" || revision == "" || file == "" || path.IsAbs(file) || strings.ContainsAny(file, "\\\x00") || path.Clean(file) != file || file == "." {
		return "", fmt.Errorf("invalid model repository, revision, or file path")
	}
	for _, part := range strings.Split(file, "/") {
		if part == ".." {
			return "", fmt.Errorf("model file path must not traverse directories")
		}
	}
	return modelStoreCheckedPath(root, "artifacts", modelArtifactID(repo, revision, file), path.Base(file))
}

func validModelDigest(value string) bool {
	decoded, err := hex.DecodeString(value)
	return err == nil && len(decoded) == sha256.Size && value == strings.ToLower(value)
}

func validateModelArtifact(root string, artifact modelArtifact) error {
	if !validModelDigest(artifact.ID) || !validModelDigest(artifact.SHA256) || artifact.Size < 0 || !filepath.IsAbs(artifact.Path) || filepath.Clean(artifact.Path) != artifact.Path {
		return fmt.Errorf("invalid model artifact registration")
	}
	if artifact.Managed || artifact.Repo != "" {
		expected, err := modelArtifactPath(root, artifact.Repo, artifact.Revision, artifact.File)
		if err != nil {
			return err
		}
		if artifact.ID != modelArtifactID(artifact.Repo, artifact.Revision, artifact.File) || (artifact.Managed && artifact.Path != expected) {
			return fmt.Errorf("model registration does not match its repository identity or owned path")
		}
	} else if artifact.ID != modelArtifactID("", "", artifact.Path) || artifact.Revision != "" || artifact.File != filepath.Base(artifact.Path) || artifact.UpstreamVerified {
		return fmt.Errorf("external model registration has invalid identity or provenance")
	}
	return nil
}

func modelRegistrationPath(root, id string) (string, error) {
	if !validModelDigest(id) {
		return "", cliError{"invalid_model_id", "A registered model ID is required.", []string{"Use lmx model list to find the ID."}, nil}
	}
	return modelStoreCheckedPath(root, "registrations", id+".json")
}

func saveModelArtifact(root string, artifact modelArtifact) error {
	if err := validateModelArtifact(root, artifact); err != nil {
		return err
	}
	registration, err := modelRegistrationPath(root, artifact.ID)
	if err != nil {
		return err
	}
	dir := filepath.Dir(registration)
	if err := os.MkdirAll(dir, 0o755); err != nil {
		return err
	}
	if _, err := modelRegistrationPath(root, artifact.ID); err != nil {
		return err
	}
	file, err := os.CreateTemp(dir, ".registration-*")
	if err != nil {
		return err
	}
	defer os.Remove(file.Name())
	defer file.Close()
	encoder := json.NewEncoder(file)
	encoder.SetIndent("", "  ")
	if err := encoder.Encode(artifact); err != nil {
		return err
	}
	if err := file.Sync(); err != nil {
		return err
	}
	if err := file.Close(); err != nil {
		return err
	}
	if err := os.Rename(file.Name(), registration); err != nil {
		return err
	}
	directory, err := os.Open(dir)
	if err != nil {
		return err
	}
	defer directory.Close()
	return directory.Sync()
}

func readModelArtifact(root, id string) (modelArtifact, error) {
	var artifact modelArtifact
	registration, err := modelRegistrationPath(root, id)
	if err != nil {
		return artifact, err
	}
	info, err := os.Lstat(registration)
	if err != nil {
		return artifact, err
	}
	if !info.Mode().IsRegular() || info.Size() > 1024*1024 {
		return artifact, fmt.Errorf("invalid model registration file: %s", registration)
	}
	data, err := os.ReadFile(registration)
	if err != nil {
		return artifact, err
	}
	if err := json.Unmarshal(data, &artifact); err != nil {
		return artifact, err
	}
	if artifact.ID != id {
		return artifact, fmt.Errorf("model registration ID does not match its filename")
	}
	return artifact, validateModelArtifact(root, artifact)
}

func modelFileDigest(filename string) (string, int64, error) {
	info, err := os.Stat(filename)
	if err != nil {
		return "", 0, err
	}
	if !info.Mode().IsRegular() {
		return "", 0, fmt.Errorf("model source must be a regular file: %s", filename)
	}
	file, err := os.Open(filename)
	if err != nil {
		return "", 0, err
	}
	defer file.Close()
	before, err := file.Stat()
	if err != nil {
		return "", 0, err
	}
	if !before.Mode().IsRegular() || !os.SameFile(info, before) {
		return "", 0, fmt.Errorf("model source changed while opening: %s", filename)
	}
	hash := sha256.New()
	size, err := io.Copy(hash, file)
	if err != nil {
		return "", 0, err
	}
	after, err := file.Stat()
	if err != nil {
		return "", 0, err
	}
	current, err := os.Stat(filename)
	if err != nil {
		return "", 0, err
	}
	if size != before.Size() || size != after.Size() || !before.ModTime().Equal(after.ModTime()) || !os.SameFile(after, current) {
		return "", 0, fmt.Errorf("model source changed while hashing: %s", filename)
	}
	return hex.EncodeToString(hash.Sum(nil)), size, nil
}

func handleModelImport(target string, args cliArgs) error {
	if target == "" {
		return cliError{"missing_model_path", "model import requires a local file path.", nil, nil}
	}
	root, err := modelStoreRoot(args)
	if err != nil {
		return err
	}
	source, err := filepath.Abs(target)
	if err != nil {
		return err
	}
	// Resolve cache symlinks, but never claim ownership of their targets.
	source, err = filepath.EvalSymlinks(source)
	if err != nil {
		return err
	}
	digest, size, err := modelFileDigest(source)
	if err != nil {
		return err
	}
	artifact := modelArtifact{ID: modelArtifactID("", "", source), File: filepath.Base(source), Path: source, SHA256: digest, Size: size, Integrity: "local-sha256"}
	if !hasFlag(args, "dry-run") {
		if err := saveModelArtifact(root, artifact); err != nil {
			return err
		}
	}
	return writeOrPrintJSON("model_import", args, map[string]any{"artifact": artifact, "dryRun": hasFlag(args, "dry-run"), "copied": false})
}

func handleModelVerify(target string, args cliArgs) error {
	root, err := modelStoreRoot(args)
	if err != nil {
		return err
	}
	artifact, err := readModelArtifact(root, target)
	if err != nil {
		return err
	}
	digest, size, err := modelFileDigest(artifact.Path)
	if err != nil {
		return cliError{"model_verification_failed", "Could not read the registered model source.", nil, err.Error()}
	}
	verified := digest == artifact.SHA256 && size == artifact.Size
	result := map[string]any{"id": artifact.ID, "path": artifact.Path, "verified": verified, "sha256": digest, "size": size, "expectedSHA256": artifact.SHA256, "expectedSize": artifact.Size}
	if err := writeOrPrintJSON("model_verify", args, result); err != nil {
		return err
	}
	if !verified {
		return cliError{"model_checksum_mismatch", "Model source no longer matches its registered checksum and size.", nil, result}
	}
	return nil
}

func acquireModelArtifactLock(filename string) (func(), error) {
	lockPath := filename + ".lock"
	lock, err := os.OpenFile(lockPath, os.O_WRONLY|os.O_CREATE|os.O_EXCL, 0o600)
	if err != nil {
		return nil, fmt.Errorf("cannot lock model artifact (another operation may be active): %w", err)
	}
	if err := lock.Close(); err != nil {
		os.Remove(lockPath)
		return nil, err
	}
	return func() { _ = os.Remove(lockPath) }, nil
}

func handleModelRemove(target string, args cliArgs) error {
	if !hasFlag(args, "yes") && !hasFlag(args, "dry-run") {
		return cliError{"confirmation_required", "model remove requires --yes; external source files are never deleted.", []string{"Use --dry-run to inspect the removal."}, nil}
	}
	root, err := modelStoreRoot(args)
	if err != nil {
		return err
	}
	artifact, err := readModelArtifact(root, target)
	if err != nil {
		return err
	}
	registration, err := modelRegistrationPath(root, target)
	if err != nil {
		return err
	}
	var releaseSnapshotLock func()
	if !hasFlag(args, "dry-run") {
		releaseSnapshotLock, err = acquireModelSnapshotMutation(root)
		if err != nil {
			return err
		}
		defer releaseSnapshotLock()
	}
	references, err := modelSnapshotArtifactReferences(root, artifact.ID)
	if err != nil {
		return err
	}
	if len(references) != 0 {
		return fmt.Errorf("model artifact is referenced by snapshots %s; remove those snapshots first", strings.Join(references, ", "))
	}
	if !hasFlag(args, "dry-run") {
		if artifact.Repo != "" {
			destination, err := modelArtifactPath(root, artifact.Repo, artifact.Revision, artifact.File)
			if err != nil {
				return err
			}
			if err := os.MkdirAll(filepath.Dir(destination), 0o755); err != nil {
				return err
			}
			if _, err := modelArtifactPath(root, artifact.Repo, artifact.Revision, artifact.File); err != nil {
				return err
			}
			unlock, err := acquireModelArtifactLock(destination)
			if err != nil {
				return err
			}
			defer unlock()
			// A completed pull may have refreshed the record before we took its lock.
			artifact, err = readModelArtifact(root, target)
			if err != nil {
				return err
			}
		}
		if artifact.Managed {
			info, err := os.Lstat(artifact.Path)
			if err != nil && !errors.Is(err, os.ErrNotExist) {
				return err
			}
			if err == nil {
				if !info.Mode().IsRegular() {
					return fmt.Errorf("refusing to remove non-regular managed artifact: %s", artifact.Path)
				}
				if err := os.Remove(artifact.Path); err != nil {
					return err
				}
			}
		}
		if err := os.Remove(registration); err != nil {
			return err
		}
	}
	return writeOrPrintJSON("model_remove", args, map[string]any{"id": artifact.ID, "path": artifact.Path, "managed": artifact.Managed, "sourcePreserved": !artifact.Managed, "dryRun": hasFlag(args, "dry-run")})
}
