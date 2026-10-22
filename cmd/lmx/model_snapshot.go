package main

import (
	"crypto/sha256"
	"encoding/hex"
	"encoding/json"
	"errors"
	"fmt"
	"net/url"
	"os"
	"path"
	"path/filepath"
	"sort"
	"strings"
	"sync"
	"time"
)

type modelSnapshotFile struct {
	modelPullSource
	Artifact *modelArtifact `json:"artifact,omitempty"`
}

type modelSnapshot struct {
	ID       string              `json:"id"`
	Repo     string              `json:"repo"`
	Revision string              `json:"revision"`
	Path     string              `json:"path"`
	State    string              `json:"state"`
	Size     int64               `json:"size"`
	Files    []modelSnapshotFile `json:"files"`
}

func modelSnapshotID(repo, revision string, files []modelSnapshotFile) string {
	names := make([]string, len(files))
	for i, file := range files {
		names[i] = file.File
	}
	data, _ := json.Marshal(struct {
		Repo, Revision string
		Files          []string
	}{repo, revision, names})
	digest := sha256.Sum256(data)
	return hex.EncodeToString(digest[:])
}

func modelSnapshotDir(root, id string) (string, error) {
	if !validModelDigest(id) {
		return "", errors.New("invalid model snapshot ID")
	}
	return modelStoreCheckedPath(root, "snapshots", id)
}

// The store-wide mutation lock orders snapshot references before artifact deletion.
// Artifact acquisition takes its existing per-artifact lock inside this lock.
func acquireModelSnapshotMutation(root string) (func(), error) {
	lock, err := modelStoreCheckedPath(root, ".snapshot-mutation")
	if err != nil {
		return nil, err
	}
	if err := os.MkdirAll(root, 0o755); err != nil {
		return nil, err
	}
	return acquireModelArtifactLock(lock)
}

func lockModelSnapshot(args cliArgs, id string) (func(), error) {
	root, err := modelStoreRoot(args)
	if err != nil {
		return nil, err
	}
	if _, err := modelSnapshotDir(root, id); err != nil {
		return nil, err
	}
	lock, err := modelStoreCheckedPath(root, "snapshots", id+".in-use.lock")
	if err != nil {
		return nil, err
	}
	if err := os.MkdirAll(filepath.Dir(lock), 0o700); err != nil {
		return nil, err
	}
	if _, err := modelStoreCheckedPath(root, "snapshots", filepath.Base(lock)); err != nil {
		return nil, err
	}
	file, err := os.OpenFile(lock, os.O_CREATE|os.O_RDWR, 0o600)
	if err != nil {
		return nil, err
	}
	if err := lockTerminalRunFile(file); err != nil {
		file.Close()
		return nil, fmt.Errorf("snapshot is locked by another operation: %w", err)
	}
	// Keep the inode, as with installation locks. The OS releases ownership on
	// process death so service restarts do not require manual stale-lock removal.
	return func() { _ = unlockTerminalRunFile(file); _ = file.Close() }, nil
}

func modelSnapshotPatterns(value string) ([]string, error) {
	if value == "" {
		return nil, nil
	}
	patterns := strings.Split(value, ",")
	for i, pattern := range patterns {
		pattern = strings.TrimSpace(pattern)
		if !modelPullSafePath(pattern) {
			return nil, fmt.Errorf("invalid snapshot glob %q", pattern)
		}
		for _, component := range strings.Split(pattern, "/") {
			if _, err := path.Match(component, ""); err != nil {
				return nil, fmt.Errorf("invalid snapshot glob %q: %w", pattern, err)
			}
		}
		patterns[i] = pattern
	}
	return patterns, nil
}

// ** spans whole directory components; other syntax follows path.Match.
func modelSnapshotGlob(pattern, name string) bool {
	parts, names := strings.Split(pattern, "/"), strings.Split(name, "/")
	var match func(int, int) bool
	match = func(p, n int) bool {
		if p == len(parts) {
			return n == len(names)
		}
		if parts[p] == "**" {
			for i := n; i <= len(names); i++ {
				if match(p+1, i) {
					return true
				}
			}
			return false
		}
		if n == len(names) {
			return false
		}
		ok, _ := path.Match(parts[p], names[n])
		return ok && match(p+1, n+1)
	}
	return match(0, 0)
}

func modelSnapshotSelected(name string, include, exclude []string) bool {
	selected := len(include) == 0
	for _, pattern := range include {
		selected = selected || modelSnapshotGlob(pattern, name)
	}
	for _, pattern := range exclude {
		if modelSnapshotGlob(pattern, name) {
			return false
		}
	}
	return selected
}

func validateModelSnapshotNames(files []modelSnapshotFile) error {
	seen := make(map[string]bool, len(files))
	for _, file := range files {
		if !modelPullSafePath(file.File) {
			return fmt.Errorf("unsafe repository filename %q", file.File)
		}
		if seen[file.File] {
			return fmt.Errorf("duplicate repository filename %q", file.File)
		}
		seen[file.File] = true
	}
	for name := range seen {
		for parent := path.Dir(name); parent != "."; parent = path.Dir(parent) {
			if seen[parent] {
				return fmt.Errorf("repository file/directory collision: %q and %q", parent, name)
			}
		}
	}
	return nil
}

func handleModelSnapshot(target string, args cliArgs) error {
	if !modelPullSafePath(target) || len(strings.Split(target, "/")) > 2 {
		return errors.New("model snapshot requires a safe HuggingFace repository name")
	}
	include, err := modelSnapshotPatterns(opt(args, "include"))
	if err != nil {
		return err
	}
	exclude, err := modelSnapshotPatterns(opt(args, "exclude"))
	if err != nil {
		return err
	}
	root, err := modelStoreRoot(args)
	if err != nil {
		return err
	}
	base, err := url.Parse(strings.TrimRight(hfAPIURL(args), "/"))
	if err != nil || base == nil || (base.Scheme != "https" && base.Scheme != "http") || base.Host == "" || base.User != nil || base.RawQuery != "" || base.Fragment != "" {
		return errors.New("invalid HuggingFace API URL")
	}
	client := modelPullHTTPClient()
	defer client.CloseIdleConnections()
	listing, revision, err := pirateFaceListingFor(args, client, target)
	if err != nil {
		return err
	}
	commit, siblings, err := resolveModelPullRevision(client, base.String(), target, revision)
	if err != nil {
		return err
	}
	snapshot := modelSnapshot{Repo: target, Revision: commit, State: "incomplete", Files: []modelSnapshotFile{}}
	allNames := make([]modelSnapshotFile, 0, len(siblings))
	for _, sibling := range siblings {
		allNames = append(allNames, modelSnapshotFile{modelPullSource: modelPullSource{File: sibling.Name}})
		if !modelSnapshotSelected(sibling.Name, include, exclude) {
			continue
		}
		source, err := modelPullSourceForFile(base.String(), target, commit, sibling)
		if err != nil {
			return fmt.Errorf("%s: %w", sibling.Name, err)
		}
		if source.ExpectedHash == "" {
			return fmt.Errorf("%s: snapshot requires an upstream file hash", sibling.Name)
		}
		if source.Size > int64(1<<63-1)-snapshot.Size {
			return errors.New("snapshot total size exceeds supported range")
		}
		snapshot.Size += source.Size
		snapshot.Files = append(snapshot.Files, modelSnapshotFile{modelPullSource: source})
	}
	if err := validateModelSnapshotNames(allNames); err != nil {
		return err
	}
	if len(snapshot.Files) == 0 {
		return errors.New("snapshot selection contains no files")
	}
	sort.Slice(snapshot.Files, func(i, j int) bool { return snapshot.Files[i].File < snapshot.Files[j].File })
	snapshot.ID = modelSnapshotID(target, commit, snapshot.Files)
	dir, err := modelSnapshotDir(root, snapshot.ID)
	if err != nil {
		return err
	}
	snapshot.Path = filepath.Join(dir, "model")
	if hasFlag(args, "dry-run") || !hasFlag(args, "yes") {
		return writeOrPrintJSON("model_snapshot", args, map[string]any{"action": "plan", "snapshot": snapshot, "include": include, "exclude": exclude, "downloadRequiresYes": true, "pirateFace": pirateFacePlan(listing)})
	}
	unlock, err := acquireModelSnapshotMutation(root)
	if err != nil {
		return err
	}
	defer func() {
		if unlock != nil {
			unlock()
		}
	}()
	if err := os.MkdirAll(filepath.Dir(dir), 0o755); err != nil {
		return err
	}
	releaseSnapshot, err := lockModelSnapshot(args, snapshot.ID)
	if err != nil {
		return err
	}
	defer releaseSnapshot()
	if existing, err := readModelSnapshot(root, snapshot.ID); err == nil {
		if existing.State == "ready" {
			if err := verifyModelSnapshot(existing, true); err != nil {
				return err
			}
			return writeOrPrintJSON("model_snapshot", args, map[string]any{"action": "already_present", "snapshot": existing})
		}
		// Retain previously recorded artifact targets when resuming owned links.
		for i := range snapshot.Files {
			snapshot.Files[i].Artifact = existing.Files[i].Artifact
		}
	} else if !errors.Is(err, os.ErrNotExist) {
		return err
	} else {
		if err := os.MkdirAll(filepath.Dir(dir), 0o755); err != nil {
			return err
		}
		if err := os.Mkdir(dir, 0o755); err != nil {
			return fmt.Errorf("refusing to adopt an unowned snapshot directory: %w", err)
		}
		if err := writeModelSnapshot(root, snapshot); err != nil {
			_ = os.Remove(dir)
			return err
		}
	}
	if err := checkModelSnapshotTree(snapshot); err != nil {
		return err
	}
	if err := os.MkdirAll(snapshot.Path, 0o755); err != nil {
		return err
	}
	// Durable pending references now protect artifacts; do not serialize unrelated
	// snapshots or artifact removals behind network downloads.
	unlock()
	unlock = nil
	ctx, stopSignals := terminalSignalContext()
	defer stopSignals()
	downloader, err := newModelDownloader(ctx, client, args)
	if err != nil {
		return err
	}
	// Files download concurrently; the downloader caps total streams. A failed
	// file leaves the snapshot incomplete while the others keep their progress.
	stopProgress := downloader.reportProgress(10 * time.Second)
	sources := make([]modelPullSource, len(snapshot.Files))
	for i, file := range snapshot.Files {
		sources[i] = file.modelPullSource
	}
	if downloader.swarm, err = runPirateFaceSwarm(downloader, root, listing, sources); err != nil {
		stopProgress()
		return fmt.Errorf("snapshot %s remains incomplete; rerun to resume: %w", snapshot.ID, err)
	}
	defer downloader.swarm.cleanup(ctx)
	acquired := make([]modelArtifact, len(snapshot.Files))
	failures := make([]error, len(snapshot.Files))
	work := make(chan int)
	var wg sync.WaitGroup
	for range min(downloader.connections, len(snapshot.Files)) {
		wg.Add(1)
		go func() {
			defer wg.Done()
			for i := range work {
				if err := ctx.Err(); err != nil {
					failures[i] = err
					continue
				}
				acquired[i], _, failures[i] = acquireModelPull(downloader, root, snapshot.Files[i].modelPullSource)
			}
		}()
	}
	for i := range snapshot.Files {
		work <- i
	}
	close(work)
	wg.Wait()
	stopProgress()
	var incomplete []error
	for i, err := range failures {
		if err != nil {
			incomplete = append(incomplete, fmt.Errorf("%s: %w", snapshot.Files[i].File, err))
		}
	}
	if len(incomplete) > 0 {
		return fmt.Errorf("snapshot %s remains incomplete; rerun to resume: %w", snapshot.ID, errors.Join(incomplete...))
	}
	for i := range snapshot.Files {
		file := &snapshot.Files[i]
		artifact := acquired[i]
		link := filepath.Join(snapshot.Path, filepath.FromSlash(file.File))
		if previous, err := os.Readlink(link); err == nil {
			if file.Artifact == nil || previous != file.Artifact.Path {
				return fmt.Errorf("refusing to replace unowned snapshot link %s", link)
			}
			if previous != artifact.Path {
				if err := os.Remove(link); err != nil {
					return err
				}
			}
		} else if !errors.Is(err, os.ErrNotExist) {
			return err
		}
		file.Artifact = &artifact
	}
	// Persist all ownership targets once before creating links. Acquired artifacts
	// are already resumable through their registrations if this phase is interrupted.
	if err := writeModelSnapshot(root, snapshot); err != nil {
		return err
	}
	for _, file := range snapshot.Files {
		link := filepath.Join(snapshot.Path, filepath.FromSlash(file.File))
		if err := os.MkdirAll(filepath.Dir(link), 0o755); err != nil {
			return err
		}
		if _, err := os.Lstat(link); errors.Is(err, os.ErrNotExist) {
			if err := os.Symlink(file.Artifact.Path, link); err != nil {
				return err
			}
		} else if err != nil {
			return err
		}
	}
	// Acquisition just verified every byte; promotion only needs to recheck links
	// and availability rather than reading an entire repository a second time.
	if err := verifyModelSnapshot(snapshot, false); err != nil {
		return err
	}
	snapshot.State = "ready"
	if err := writeModelSnapshot(root, snapshot); err != nil {
		return err
	}
	if err := os.Remove(filepath.Join(dir, "pending.json")); err != nil && !errors.Is(err, os.ErrNotExist) {
		return err
	}
	return writeOrPrintJSON("model_snapshot", args, map[string]any{"action": "ready", "snapshot": snapshot})
}

func validateModelSnapshot(root string, snapshot modelSnapshot) error {
	dir, err := modelSnapshotDir(root, snapshot.ID)
	if err != nil {
		return err
	}
	if snapshot.Path != filepath.Join(dir, "model") || !modelPullSafePath(snapshot.Repo) || len(strings.Split(snapshot.Repo, "/")) > 2 || !modelPullHex(snapshot.Revision, 20) || (snapshot.State != "ready" && snapshot.State != "incomplete") || len(snapshot.Files) == 0 {
		return errors.New("invalid snapshot identity or state")
	}
	if err := validateModelSnapshotNames(snapshot.Files); err != nil {
		return err
	}
	var size int64
	for i, file := range snapshot.Files {
		if file.Repo != snapshot.Repo || file.Revision != snapshot.Revision || file.Size < 0 || file.Size > int64(1<<63-1)-size || (i > 0 && snapshot.Files[i-1].File >= file.File) {
			return errors.New("invalid snapshot file provenance or ordering")
		}
		if (file.Integrity != "sha256" || !validModelDigest(file.ExpectedHash)) && (file.Integrity != "git-blob-sha1" || !modelPullHex(file.ExpectedHash, 20)) {
			return errors.New("invalid snapshot upstream hash")
		}
		size += file.Size
		if file.Artifact == nil {
			if snapshot.State == "ready" {
				return errors.New("ready snapshot lacks acquired artifact")
			}
			continue
		}
		a := file.Artifact
		if err := validateModelArtifact(root, *a); err != nil {
			return err
		}
		if a.Repo != file.Repo || a.Revision != file.Revision || a.File != file.File || a.Size != file.Size || a.Integrity != file.Integrity || a.ExpectedHash != file.ExpectedHash || !a.UpstreamVerified {
			return errors.New("snapshot artifact differs from pinned provenance")
		}
	}
	if size != snapshot.Size || snapshot.ID != modelSnapshotID(snapshot.Repo, snapshot.Revision, snapshot.Files) {
		return errors.New("snapshot manifest identity or total size mismatch")
	}
	return nil
}

func writeModelSnapshot(root string, snapshot modelSnapshot) error {
	if err := validateModelSnapshot(root, snapshot); err != nil {
		return err
	}
	name := "pending.json"
	if snapshot.State == "ready" {
		name = "manifest.json"
	}
	filename, err := modelStoreCheckedPath(root, "snapshots", snapshot.ID, name)
	if err != nil {
		return err
	}
	file, err := os.CreateTemp(filepath.Dir(filename), ".manifest-*")
	if err != nil {
		return err
	}
	defer os.Remove(file.Name())
	defer file.Close()
	encoder := json.NewEncoder(file)
	encoder.SetIndent("", "  ")
	if err := encoder.Encode(snapshot); err != nil {
		return err
	}
	if err := file.Sync(); err != nil {
		return err
	}
	if err := file.Close(); err != nil {
		return err
	}
	if err := os.Rename(file.Name(), filename); err != nil {
		return err
	}
	dir, err := os.Open(filepath.Dir(filename))
	if err != nil {
		return err
	}
	defer dir.Close()
	return dir.Sync()
}

func readModelSnapshot(root, id string) (modelSnapshot, error) {
	var snapshot modelSnapshot
	if _, err := modelSnapshotDir(root, id); err != nil {
		return snapshot, err
	}
	for _, name := range []string{"manifest.json", "pending.json"} {
		filename, err := modelStoreCheckedPath(root, "snapshots", id, name)
		if err != nil {
			return snapshot, err
		}
		info, err := os.Lstat(filename)
		if errors.Is(err, os.ErrNotExist) {
			continue
		}
		if err != nil {
			return snapshot, err
		}
		if !info.Mode().IsRegular() || info.Size() > 128<<20 {
			return snapshot, errors.New("invalid snapshot manifest file")
		}
		data, err := os.ReadFile(filename)
		if err != nil {
			return snapshot, err
		}
		if err := json.Unmarshal(data, &snapshot); err != nil {
			return snapshot, err
		}
		if snapshot.ID != id || (name == "manifest.json") != (snapshot.State == "ready") {
			return snapshot, errors.New("snapshot manifest name/state mismatch")
		}
		return snapshot, validateModelSnapshot(root, snapshot)
	}
	return snapshot, os.ErrNotExist
}

// Only expected directories and exact owned links may occur in the model tree.
// This also prevents upstream names from colliding with our out-of-tree metadata.
func checkModelSnapshotTree(snapshot modelSnapshot) error {
	allowed := map[string]*modelArtifact{}
	for _, file := range snapshot.Files {
		allowed[file.File] = file.Artifact
		for parent := path.Dir(file.File); parent != "."; parent = path.Dir(parent) {
			allowed[parent] = nil
		}
	}
	return filepath.WalkDir(snapshot.Path, func(filename string, entry os.DirEntry, walkErr error) error {
		if errors.Is(walkErr, os.ErrNotExist) && filename == snapshot.Path {
			return nil
		}
		if walkErr != nil {
			return walkErr
		}
		if filename == snapshot.Path {
			if !entry.IsDir() {
				return errors.New("snapshot model path is not a directory")
			}
			return nil
		}
		relative, err := filepath.Rel(snapshot.Path, filename)
		if err != nil {
			return err
		}
		artifact, ok := allowed[filepath.ToSlash(relative)]
		if !ok {
			return fmt.Errorf("unowned path in snapshot directory: %s", filename)
		}
		if entry.IsDir() && artifact == nil {
			return nil
		}
		if artifact == nil || entry.Type()&os.ModeSymlink == 0 {
			return fmt.Errorf("snapshot path is not an owned link: %s", filename)
		}
		target, err := os.Readlink(filename)
		if err != nil {
			return err
		}
		if target != artifact.Path {
			return fmt.Errorf("snapshot link target changed: %s", filename)
		}
		return nil
	})
}

func verifyModelSnapshot(snapshot modelSnapshot, hashes bool) error {
	if err := checkModelSnapshotTree(snapshot); err != nil {
		return err
	}
	for _, file := range snapshot.Files {
		if file.Artifact == nil {
			return fmt.Errorf("snapshot file has not been acquired: %s", file.File)
		}
		filename := filepath.Join(snapshot.Path, filepath.FromSlash(file.File))
		target, err := os.Readlink(filename)
		if err != nil {
			return err
		}
		if target != file.Artifact.Path {
			return fmt.Errorf("snapshot file link changed: %s", file.File)
		}
		info, err := os.Stat(filename)
		if err != nil {
			return err
		}
		if !info.Mode().IsRegular() || info.Size() != file.Size {
			return fmt.Errorf("snapshot file size/type mismatch: %s", file.File)
		}
		if hashes {
			digest, err := verifyModelPullFile(filename, file.modelPullSource)
			if err != nil {
				return fmt.Errorf("snapshot file %s: %w", file.File, err)
			}
			if digest != file.Artifact.SHA256 {
				return fmt.Errorf("snapshot local SHA256 mismatch: %s", file.File)
			}
		}
	}
	return nil
}

func loadModelSnapshot(args cliArgs, id string) (modelSnapshot, error) {
	root, err := modelStoreRoot(args)
	if err != nil {
		return modelSnapshot{}, err
	}
	snapshot, err := readModelSnapshot(root, id)
	if err != nil {
		return snapshot, err
	}
	if snapshot.State != "ready" {
		return snapshot, errors.New("snapshot is incomplete; rerun model snapshot with its pinned revision and selection")
	}
	return snapshot, verifyModelSnapshot(snapshot, false)
}

func modelSnapshotInventory(root string) ([]modelSnapshot, error) {
	dir, err := modelStoreCheckedPath(root, "snapshots")
	if err != nil {
		return nil, err
	}
	entries, err := os.ReadDir(dir)
	if err != nil && !errors.Is(err, os.ErrNotExist) {
		return nil, err
	}
	snapshots := make([]modelSnapshot, 0, len(entries))
	for _, entry := range entries {
		if !validModelDigest(entry.Name()) {
			continue
		}
		snapshot, err := readModelSnapshot(root, entry.Name())
		if err != nil {
			return nil, fmt.Errorf("snapshot %s: %w", entry.Name(), err)
		}
		snapshots = append(snapshots, snapshot)
	}
	return snapshots, nil
}

func modelSnapshotArtifactReferences(root, artifactID string) ([]string, error) {
	snapshots, err := modelSnapshotInventory(root)
	if err != nil {
		return nil, err
	}
	var ids []string
	for _, snapshot := range snapshots {
		for _, file := range snapshot.Files {
			if modelArtifactID(file.Repo, file.Revision, file.File) == artifactID {
				ids = append(ids, snapshot.ID)
				break
			}
		}
	}
	return ids, nil
}

func handleModelSnapshots(action, target string, args cliArgs) error {
	root, err := modelStoreRoot(args)
	if err != nil {
		return err
	}
	if action == "list" {
		snapshots, err := modelSnapshotInventory(root)
		if err != nil {
			return err
		}
		type listedSnapshot struct {
			modelSnapshot
			Status      string `json:"status"`
			SourceError string `json:"sourceError,omitempty"`
		}
		listed := make([]listedSnapshot, 0, len(snapshots))
		for _, snapshot := range snapshots {
			item := listedSnapshot{modelSnapshot: snapshot, Status: snapshot.State}
			if snapshot.State == "ready" {
				if err := verifyModelSnapshot(snapshot, false); err != nil {
					item.Status = "unavailable"
					item.SourceError = err.Error()
				}
			}
			listed = append(listed, item)
		}
		return writeOrPrintJSON("model_snapshots", args, map[string]any{"root": root, "snapshots": listed})
	}
	if action != "verify" && action != "remove" {
		return fmt.Errorf("unknown snapshot action %q", action)
	}
	if action == "remove" && !hasFlag(args, "yes") && !hasFlag(args, "dry-run") {
		return errors.New("model snapshots remove requires --yes; shared artifacts and cache files are preserved")
	}
	if action == "remove" && !hasFlag(args, "dry-run") {
		unlock, err := acquireModelSnapshotMutation(root)
		if err != nil {
			return err
		}
		defer unlock()
		releaseSnapshot, err := lockModelSnapshot(args, target)
		if err != nil {
			return err
		}
		defer releaseSnapshot()
	}
	snapshot, err := readModelSnapshot(root, target)
	if err != nil {
		return err
	}
	if action == "verify" {
		verifyErr := verifyModelSnapshot(snapshot, true)
		if snapshot.State != "ready" && verifyErr == nil {
			verifyErr = errors.New("snapshot is incomplete")
		}
		result := map[string]any{"id": snapshot.ID, "verified": verifyErr == nil, "snapshot": snapshot}
		if verifyErr != nil {
			result["error"] = verifyErr.Error()
		}
		if err := writeOrPrintJSON("model_snapshot_verify", args, result); err != nil {
			return err
		}
		return verifyErr
	}
	if !hasFlag(args, "dry-run") {
		if err := removeModelSnapshot(root, snapshot); err != nil {
			return err
		}
	}
	return writeOrPrintJSON("model_snapshot_remove", args, map[string]any{"id": snapshot.ID, "sharedArtifactsPreserved": true, "dryRun": hasFlag(args, "dry-run")})
}

func removeModelSnapshot(root string, snapshot modelSnapshot) error {
	if err := checkModelSnapshotTree(snapshot); err != nil {
		return err
	}
	dir, err := modelSnapshotDir(root, snapshot.ID)
	if err != nil {
		return err
	}
	// Refuse unexpected outer files before touching any owned links or manifests.
	entries, err := os.ReadDir(dir)
	if err != nil {
		return err
	}
	for _, entry := range entries {
		if entry.Name() != "model" && entry.Name() != "manifest.json" && entry.Name() != "pending.json" {
			return fmt.Errorf("unowned file in snapshot directory: %s", entry.Name())
		}
	}
	var paths []string
	err = filepath.WalkDir(snapshot.Path, func(filename string, _ os.DirEntry, walkErr error) error {
		if errors.Is(walkErr, os.ErrNotExist) && filename == snapshot.Path {
			return nil
		}
		if walkErr != nil {
			return walkErr
		}
		paths = append(paths, filename)
		return nil
	})
	if err != nil {
		return err
	}
	for i := len(paths) - 1; i >= 0; i-- {
		if err := os.Remove(paths[i]); err != nil {
			return err
		}
	}
	for _, name := range []string{"manifest.json", "pending.json"} {
		filename, err := modelStoreCheckedPath(root, "snapshots", snapshot.ID, name)
		if err != nil {
			return err
		}
		if err := os.Remove(filename); err != nil && !errors.Is(err, os.ErrNotExist) {
			return err
		}
	}
	return os.Remove(dir)
}
