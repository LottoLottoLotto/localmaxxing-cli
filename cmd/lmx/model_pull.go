package main

import (
	"context"
	"crypto/sha1"
	"crypto/sha256"
	"encoding/hex"
	"encoding/json"
	"errors"
	"fmt"
	"hash"
	"io"
	"net/http"
	"net/url"
	"os"
	"path/filepath"
	"strconv"
	"strings"
	"sync/atomic"
	"time"
)

type modelPullFile struct {
	Name   string `json:"rfilename"`
	Size   *int64 `json:"size"`
	BlobID string `json:"blobId"`
	LFS    *struct {
		SHA256 string `json:"sha256"`
		Size   *int64 `json:"size"`
	} `json:"lfs"`
}

type modelPullSource struct {
	Repo         string `json:"repo"`
	Revision     string `json:"revision"`
	File         string `json:"file"`
	URL          string `json:"url"`
	Size         int64  `json:"size"`
	Integrity    string `json:"integrity"`
	ExpectedHash string `json:"expectedHash,omitempty"`
}

func handleModelPull(target string, args cliArgs) error {
	file := opt(args, "file")
	if !modelPullSafePath(target) || len(strings.Split(target, "/")) > 2 || !modelPullSafePath(file) {
		return errors.New("model pull requires a repository and an exact safe relative --file path (not a directory)")
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
	source, err := resolveModelPull(client, base.String(), target, file, revision)
	if err != nil {
		return err
	}
	destination, err := modelArtifactPath(root, target, source.Revision, file)
	if err != nil {
		return err
	}
	plan := map[string]any{"action": "plan", "source": source, "destination": destination, "downloadRequiresYes": true, "upstreamVerified": false, "pirateFace": pirateFacePlan(listing)}
	if hasFlag(args, "dry-run") || !hasFlag(args, "yes") {
		return writeOrPrintJSON("model_pull", args, plan)
	}
	ctx, stopSignals := terminalSignalContext()
	defer stopSignals()
	downloader, err := newModelDownloader(ctx, client, args)
	if err != nil {
		return err
	}
	stopProgress := downloader.reportProgress(10 * time.Second)
	if downloader.swarm, err = runPirateFaceSwarm(downloader, root, listing, []modelPullSource{source}); err != nil {
		stopProgress()
		return err
	}
	defer downloader.swarm.cleanup(ctx)
	artifact, action, err := acquireModelPull(downloader, root, source)
	stopProgress()
	if err != nil {
		return err
	}
	if err := writeOrPrintJSON("model_pull", args, map[string]any{"action": action, "source": source, "artifact": artifact}); err != nil {
		return err
	}
	return seedAfterDownload(args, root, listing, map[string]modelArtifact{source.File: artifact})
}

// acquireModelPull is shared by single-file pulls and full repository snapshots.
// Registration only follows complete verification; interrupted bytes stay resumable.
func acquireModelPull(downloader *modelDownloader, root string, source modelPullSource) (modelArtifact, string, error) {
	var empty modelArtifact
	destination, err := modelArtifactPath(root, source.Repo, source.Revision, source.File)
	if err != nil {
		return empty, "", err
	}
	if err := os.MkdirAll(filepath.Dir(destination), 0o755); err != nil {
		return empty, "", err
	}
	unlock, err := acquireModelArtifactLock(destination)
	if err != nil {
		return empty, "", err
	}
	defer unlock()

	register := func(path, digest, action string, managed bool) (modelArtifact, string, error) {
		artifact := modelArtifact{ID: modelArtifactID(source.Repo, source.Revision, source.File), Repo: source.Repo, Revision: source.Revision, File: source.File, Path: path, SHA256: digest, Size: source.Size, Managed: managed, Integrity: source.Integrity, ExpectedHash: source.ExpectedHash, UpstreamVerified: source.ExpectedHash != ""}
		if err := saveModelArtifact(root, artifact); err != nil {
			return empty, "", err
		}
		return artifact, action, nil
	}
	if info, err := os.Lstat(destination); err == nil {
		if !info.Mode().IsRegular() {
			return empty, "", errors.New("artifact destination is not a regular file")
		}
		digest, err := verifyModelPullFile(destination, source)
		if err != nil {
			return empty, "", fmt.Errorf("existing artifact failed verification: %w", err)
		}
		return register(destination, digest, "already_present", true)
	} else if !os.IsNotExist(err) {
		return empty, "", err
	}

	// Cache references remain external: removal unregisters them, never deletes them.
	if source.ExpectedHash != "" {
		for _, candidate := range modelPullCachePaths(source) {
			canonical, err := filepath.EvalSymlinks(candidate)
			if err != nil {
				continue
			}
			canonical, err = filepath.Abs(canonical)
			if err != nil {
				continue
			}
			digest, err := verifyModelPullFile(canonical, source)
			if err == nil {
				return register(canonical, digest, "cache_reused", false)
			}
		}
	}
	part := destination + ".part"
	if digest, ok := downloader.swarm.take(downloader.args, source, part); ok {
		if err := os.Rename(part, destination); err != nil {
			return empty, "", err
		}
		return register(destination, digest, "downloaded_pirateface", true)
	}
	if err := downloader.fetch(source, part); err != nil {
		return empty, "", err
	}
	digest, err := verifyModelPullFile(part, source)
	if err != nil {
		// A complete but corrupt partial cannot be used as the next resume prefix.
		os.Remove(part)
		os.Remove(part + ".chunks")
		return empty, "", err
	}
	if err := os.Rename(part, destination); err != nil {
		return empty, "", err
	}
	return register(destination, digest, "downloaded", true)
}

func modelPullSafePath(value string) bool {
	if value == "" || strings.ContainsAny(value, "\\\x00:") || strings.HasPrefix(value, "/") {
		return false
	}
	for _, component := range strings.Split(value, "/") {
		if component == "" || component == "." || component == ".." {
			return false
		}
	}
	return true
}

func modelPullHex(value string, bytes int) bool {
	if len(value) != bytes*2 {
		return false
	}
	_, err := hex.DecodeString(value)
	return err == nil
}

func modelPullHTTPClient() *http.Client {
	transport := http.DefaultTransport.(*http.Transport).Clone()
	transport.ResponseHeaderTimeout = 30 * time.Second
	return &http.Client{Transport: transport, CheckRedirect: func(req *http.Request, via []*http.Request) error {
		if len(via) >= 10 {
			return errors.New("too many HuggingFace redirects")
		}
		// net/http's default permits credentials on subdomains; require exact origin.
		previous := via[len(via)-1]
		if req.URL.Host != previous.URL.Host || req.URL.Scheme != previous.URL.Scheme || previous.Header.Get("Authorization") == "" {
			req.Header.Del("Authorization")
		}
		if req.URL.Scheme != "https" && req.URL.Scheme != "http" {
			return errors.New("unsupported download redirect scheme")
		}
		if previous.URL.Scheme == "https" && req.URL.Scheme != "https" {
			return errors.New("refusing insecure download redirect")
		}
		return nil
	}}
}

func modelPullRequest(ctx context.Context, endpoint string) (*http.Request, error) {
	req, err := http.NewRequestWithContext(ctx, http.MethodGet, endpoint, nil)
	if err != nil {
		return nil, err
	}
	if token, _ := huggingFaceToken(); token != "" {
		req.Header.Set("Authorization", "Bearer "+token)
	}
	req.Header.Set("Accept-Encoding", "identity")
	return req, nil
}

func resolveModelPull(client *http.Client, base, repo, file, revision string) (modelPullSource, error) {
	commit, files, err := resolveModelPullRevision(client, base, repo, revision)
	if err != nil {
		return modelPullSource{}, err
	}
	for _, sibling := range files {
		if sibling.Name == file {
			return modelPullSourceForFile(base, repo, commit, sibling)
		}
	}
	return modelPullSource{}, fmt.Errorf("file %q is not present in repository revision (directories are not downloadable)", file)
}

func resolveModelPullRevision(client *http.Client, base, repo, revision string) (string, []modelPullFile, error) {
	if revision == "" {
		revision = "main"
	}
	ctx, cancel := context.WithTimeout(context.Background(), 30*time.Second)
	defer cancel()
	req, err := modelPullRequest(ctx, base+"/api/models/"+hfRepoPath(repo)+"/revision/"+url.PathEscape(revision)+"?blobs=true")
	if err != nil {
		return "", nil, err
	}
	response, err := client.Do(req)
	if err != nil {
		return "", nil, err
	}
	defer response.Body.Close()
	if response.StatusCode != http.StatusOK {
		return "", nil, fmt.Errorf("HuggingFace revision metadata returned HTTP %d", response.StatusCode)
	}
	var metadata struct {
		SHA      string          `json:"sha"`
		Siblings []modelPullFile `json:"siblings"`
	}
	if err := json.NewDecoder(io.LimitReader(response.Body, 32<<20)).Decode(&metadata); err != nil {
		return "", nil, fmt.Errorf("invalid HuggingFace metadata: %w", err)
	}
	if !modelPullHex(metadata.SHA, 20) {
		return "", nil, errors.New("HuggingFace did not resolve revision to a full 40-hex commit")
	}
	return strings.ToLower(metadata.SHA), metadata.Siblings, nil
}

func modelPullSourceForFile(base, repo, commit string, sibling modelPullFile) (modelPullSource, error) {
	var source modelPullSource
	size := sibling.Size
	integrity, expected := "local-sha256", ""
	if sibling.LFS != nil {
		if sibling.LFS.Size != nil {
			size = sibling.LFS.Size
		}
		if sibling.LFS.SHA256 != "" {
			if !modelPullHex(sibling.LFS.SHA256, 32) {
				return source, errors.New("invalid upstream LFS SHA256")
			}
			integrity, expected = "sha256", strings.ToLower(sibling.LFS.SHA256)
		}
	} else if sibling.BlobID != "" {
		if !modelPullHex(sibling.BlobID, 20) {
			return source, errors.New("invalid upstream Git blob hash")
		}
		integrity, expected = "git-blob-sha1", strings.ToLower(sibling.BlobID)
	}
	if size == nil || *size < 0 || *size == int64(1<<63-1) {
		return source, errors.New("upstream file size is missing or invalid")
	}
	return modelPullSource{Repo: repo, Revision: commit, File: sibling.Name, URL: base + "/" + hfRepoPath(repo) + "/resolve/" + commit + "/" + hfRepoPath(sibling.Name), Size: *size, Integrity: integrity, ExpectedHash: expected}, nil
}

func verifyModelPullFile(path string, source modelPullSource) (string, error) {
	file, err := os.Open(path)
	if err != nil {
		return "", err
	}
	defer file.Close()
	info, err := file.Stat()
	if err != nil {
		return "", err
	}
	if !info.Mode().IsRegular() || info.Size() != source.Size {
		return "", fmt.Errorf("file size/type mismatch: expected a regular file of %d bytes", source.Size)
	}
	local := sha256.New()
	var upstream hash.Hash
	var sink io.Writer = local
	if source.Integrity == "git-blob-sha1" {
		upstream = sha1.New()
		fmt.Fprintf(upstream, "blob %d\x00", source.Size)
		sink = io.MultiWriter(local, upstream)
	}
	n, err := io.Copy(sink, io.LimitReader(file, source.Size+1))
	if err != nil {
		return "", err
	}
	if n != source.Size {
		return "", errors.New("file changed during integrity verification")
	}
	digest := hex.EncodeToString(local.Sum(nil))
	actual := digest
	if upstream != nil {
		actual = hex.EncodeToString(upstream.Sum(nil))
	}
	if source.ExpectedHash != "" && actual != source.ExpectedHash {
		return "", fmt.Errorf("upstream %s integrity mismatch: expected %s, got %s", source.Integrity, source.ExpectedHash, actual)
	}
	return digest, nil
}

func downloadModelPull(client *http.Client, source modelPullSource, part string) error {
	return downloadModelPullCounted(context.Background(), client, source, part, nil)
}

// downloadModelPullCounted is the single-stream download; received, when set,
// counts bytes written for progress reporting.
func downloadModelPullCounted(ctx context.Context, client *http.Client, source modelPullSource, part string, received *atomic.Int64) error {
	var offset int64
	if info, err := os.Lstat(part); err == nil {
		if !info.Mode().IsRegular() {
			return errors.New("partial download is not a regular file")
		}
		offset = info.Size()
		if offset > source.Size {
			return errors.New("partial download exceeds expected file size; remove the managed .part file before retrying")
		}
		if offset == source.Size {
			return nil
		}
	} else if !os.IsNotExist(err) {
		return err
	}
	req, err := modelPullRequest(ctx, source.URL)
	if err != nil {
		return err
	}
	if offset > 0 {
		req.Header.Set("Range", fmt.Sprintf("bytes=%d-", offset))
	}
	response, err := client.Do(req)
	if err != nil {
		return err
	}
	defer response.Body.Close()
	if response.Header.Get("Content-Encoding") != "" && response.Header.Get("Content-Encoding") != "identity" {
		return errors.New("unexpected content encoding for model download")
	}
	switch response.StatusCode {
	case http.StatusPartialContent:
		if !modelPullContentRange(response.Header.Get("Content-Range"), offset, source.Size) {
			return errors.New("invalid Content-Range for pinned model download")
		}
	case http.StatusOK:
		// Servers may ignore Range. Restart rather than append a full response.
		offset = 0
	default:
		return fmt.Errorf("HuggingFace download returned HTTP %d", response.StatusCode)
	}
	remaining := source.Size - offset
	if response.ContentLength >= 0 && response.ContentLength != remaining {
		return fmt.Errorf("download Content-Length mismatch: expected %d, got %d", remaining, response.ContentLength)
	}
	flags := os.O_WRONLY | os.O_CREATE
	if offset == 0 {
		flags |= os.O_TRUNC
	}
	file, err := os.OpenFile(part, flags, 0o600)
	if err != nil {
		return err
	}
	defer file.Close()
	if _, err := file.Seek(offset, io.SeekStart); err != nil {
		return err
	}
	var destination io.Writer = file
	if received != nil {
		destination = countingWriter{file, received}
	}
	n, copyErr := io.Copy(destination, io.LimitReader(response.Body, remaining+1))
	if n > remaining {
		file.Close()
		os.Remove(part)
		return errors.New("download exceeds expected file size")
	}
	if copyErr != nil {
		return fmt.Errorf("download interrupted; partial retained for retry: %w", copyErr)
	}
	if n != remaining {
		return fmt.Errorf("download truncated: expected %d bytes, received %d; partial retained for retry", remaining, n)
	}
	if err := file.Sync(); err != nil {
		return err
	}
	return file.Close()
}

type countingWriter struct {
	io.Writer
	count *atomic.Int64
}

func (w countingWriter) Write(p []byte) (int, error) {
	n, err := w.Writer.Write(p)
	w.count.Add(int64(n))
	return n, err
}

func modelPullContentRange(value string, offset, total int64) bool {
	return modelPullContentRangeSpan(value, offset, total-1, total)
}

// modelPullContentRangeSpan checks "bytes start-end/total" exactly.
func modelPullContentRangeSpan(value string, wantStart, wantEnd, total int64) bool {
	if !strings.HasPrefix(value, "bytes ") {
		return false
	}
	parts := strings.Split(strings.TrimPrefix(value, "bytes "), "/")
	if len(parts) != 2 {
		return false
	}
	bounds := strings.Split(parts[0], "-")
	if len(bounds) != 2 {
		return false
	}
	start, err1 := strconv.ParseInt(bounds[0], 10, 64)
	end, err2 := strconv.ParseInt(bounds[1], 10, 64)
	size, err3 := strconv.ParseInt(parts[1], 10, 64)
	return err1 == nil && err2 == nil && err3 == nil && start == wantStart && end == wantEnd && size == total && start <= end
}

func modelPullCachePaths(source modelPullSource) []string {
	cache := os.Getenv("HF_HUB_CACHE")
	if cache == "" {
		home := huggingFaceHome()
		if home == "" {
			return nil
		}
		cache = filepath.Join(home, "hub")
	}
	repo := filepath.Join(cache, "models--"+strings.ReplaceAll(source.Repo, "/", "--"))
	paths := []string{filepath.Join(repo, "snapshots", source.Revision, filepath.FromSlash(source.File))}
	if source.ExpectedHash != "" {
		paths = append(paths, filepath.Join(repo, "blobs", source.ExpectedHash))
	}
	return paths
}
