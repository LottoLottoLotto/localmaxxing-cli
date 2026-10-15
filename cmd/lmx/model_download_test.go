package main

import (
	"bytes"
	"context"
	"crypto/rand"
	"crypto/sha256"
	"encoding/hex"
	"encoding/json"
	"net/http"
	"net/http/httptest"
	"os"
	"path/filepath"
	"strings"
	"sync"
	"sync/atomic"
	"testing"
	"time"
)

func smallDownloadChunks(t *testing.T) {
	t.Helper()
	chunk, sync, delay := modelDownloadMinChunk, modelDownloadSyncEvery, modelDownloadRetryDelay
	modelDownloadMinChunk, modelDownloadSyncEvery, modelDownloadRetryDelay = 64<<10, 16<<10, time.Millisecond
	t.Cleanup(func() { modelDownloadMinChunk, modelDownloadSyncEvery, modelDownloadRetryDelay = chunk, sync, delay })
}

func randomModelBytes(t *testing.T, size int) []byte {
	t.Helper()
	data := make([]byte, size)
	if _, err := rand.Read(data); err != nil {
		t.Fatal(err)
	}
	return data
}

func modelDownloadTestSource(data []byte, url string) modelPullSource {
	digest := sha256.Sum256(data)
	return modelPullSource{Repo: "owner/model", Revision: strings.Repeat("a", 40), File: "model.gguf", URL: url, Size: int64(len(data)), Integrity: "sha256", ExpectedHash: hex.EncodeToString(digest[:])}
}

// abortAfter lets the first `budget` bytes through, then drops the connection.
type abortAfter struct {
	http.ResponseWriter
	budget int
}

func (w *abortAfter) Write(p []byte) (int, error) {
	if len(p) > w.budget {
		w.ResponseWriter.Write(p[:w.budget])
		w.ResponseWriter.(http.Flusher).Flush()
		panic(http.ErrAbortHandler)
	}
	w.budget -= len(p)
	return w.ResponseWriter.Write(p)
}

func newTestDownloader(t *testing.T, connections string) *modelDownloader {
	t.Helper()
	client := modelPullHTTPClient()
	t.Cleanup(client.CloseIdleConnections)
	d, err := newModelDownloader(context.Background(), client, cliArgs{opts: map[string]string{"download-connections": connections}, flags: map[string]bool{"quiet": true}})
	if err != nil {
		t.Fatal(err)
	}
	return d
}

// Large files split into concurrent byte ranges that reassemble exactly.
func TestModelDownloadParallelRanges(t *testing.T) {
	smallDownloadChunks(t)
	data := randomModelBytes(t, 1<<20+12345)
	var active, peak, ranged atomic.Int64
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if r.Header.Get("Range") != "" {
			ranged.Add(1)
		}
		now := active.Add(1)
		defer active.Add(-1)
		for old := peak.Load(); now > old && !peak.CompareAndSwap(old, now); old = peak.Load() {
		}
		time.Sleep(5 * time.Millisecond)
		http.ServeContent(w, r, "model.gguf", time.Time{}, bytes.NewReader(data))
	}))
	defer server.Close()
	source := modelDownloadTestSource(data, server.URL)
	part := filepath.Join(t.TempDir(), "model.gguf.part")
	if err := newTestDownloader(t, "4").fetch(source, part); err != nil {
		t.Fatal(err)
	}
	if _, err := verifyModelPullFile(part, source); err != nil {
		t.Fatal(err)
	}
	if _, err := os.Stat(part + ".chunks"); !os.IsNotExist(err) {
		t.Fatalf("progress file left after completion: %v", err)
	}
	if ranged.Load() < 4 || peak.Load() < 2 || peak.Load() > 4 {
		t.Fatalf("range requests = %d, peak concurrency = %d (want ≥4 ranges, 2..4 concurrent)", ranged.Load(), peak.Load())
	}
}

// An interrupted parallel download keeps its progress and resumes only the
// missing ranges; nothing already on disk is fetched again.
func TestModelDownloadParallelResumesMissingRanges(t *testing.T) {
	smallDownloadChunks(t)
	data := randomModelBytes(t, 1<<20)
	// Bytes are attributed to the phase in which their request started, since
	// handlers cancelled by the first run may still be writing.
	var mu sync.Mutex
	served := map[bool]int{}
	failing := true
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		mu.Lock()
		fail := failing
		mu.Unlock()
		writer := http.ResponseWriter(&countingResponse{ResponseWriter: w, mu: &mu, total: served, phase: fail})
		// Every attempt at the first range (including retries from its recorded
		// progress) dies after 20 KiB, so the run fails with progress on disk.
		var start int64
		if fail && r.Header.Get("Range") != "" {
			if _, err := fmtSscanRange(r.Header.Get("Range"), &start); err == nil && start < 64<<10 {
				writer = &abortAfter{ResponseWriter: writer, budget: 20 << 10}
			}
		}
		http.ServeContent(writer, r, "model.gguf", time.Time{}, bytes.NewReader(data))
	}))
	defer server.Close()
	source := modelDownloadTestSource(data, server.URL)
	part := filepath.Join(t.TempDir(), "model.gguf.part")
	if err := newTestDownloader(t, "4").fetch(source, part); err == nil {
		t.Fatal("aborted range was accepted")
	}
	raw, err := os.ReadFile(part + ".chunks")
	if err != nil {
		t.Fatalf("progress not retained after interruption: %v", err)
	}
	var state modelDownloadState
	if err := json.Unmarshal(raw, &state); err != nil {
		t.Fatal(err)
	}
	if state.Chunks[0].Done < 16<<10 {
		t.Fatalf("interrupted range recorded %d durable bytes, want ≥ one sync interval", state.Chunks[0].Done)
	}
	durable := int64(0)
	for _, c := range state.Chunks {
		durable += c.Done
	}
	mu.Lock()
	failing = false
	mu.Unlock()
	if err := newTestDownloader(t, "4").fetch(source, part); err != nil {
		t.Fatal(err)
	}
	if _, err := verifyModelPullFile(part, source); err != nil {
		t.Fatal(err)
	}
	mu.Lock()
	resumed := served[false]
	mu.Unlock()
	if want := int64(len(data)) - durable; int64(resumed) != want {
		t.Fatalf("resume transferred %d bytes, want exactly the %d missing", resumed, want)
	}
}

type countingResponse struct {
	http.ResponseWriter
	mu    *sync.Mutex
	total map[bool]int
	phase bool
}

func (w *countingResponse) Write(p []byte) (int, error) {
	n, err := w.ResponseWriter.Write(p)
	w.mu.Lock()
	w.total[w.phase] += n
	w.mu.Unlock()
	return n, err
}

func (w *countingResponse) Flush() { w.ResponseWriter.(http.Flusher).Flush() }

// A server that ignores Range falls back to one stream instead of corrupting the file.
func TestModelDownloadFallsBackWhenRangesIgnored(t *testing.T) {
	smallDownloadChunks(t)
	data := randomModelBytes(t, 512<<10)
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.Write(data)
	}))
	defer server.Close()
	source := modelDownloadTestSource(data, server.URL)
	part := filepath.Join(t.TempDir(), "model.gguf.part")
	if err := newTestDownloader(t, "4").fetch(source, part); err != nil {
		t.Fatal(err)
	}
	if _, err := verifyModelPullFile(part, source); err != nil {
		t.Fatal(err)
	}
}

// A .part left by the single-stream downloader is a valid prefix: the parallel
// path continues after it without re-requesting those bytes.
func TestModelDownloadContinuesSequentialPrefix(t *testing.T) {
	smallDownloadChunks(t)
	data := randomModelBytes(t, 1<<20)
	prefix := 300 << 10
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		var start int64
		if _, err := fmtSscanRange(r.Header.Get("Range"), &start); err != nil || start < int64(prefix) {
			t.Errorf("request for already-downloaded bytes: %q", r.Header.Get("Range"))
		}
		http.ServeContent(w, r, "model.gguf", time.Time{}, bytes.NewReader(data))
	}))
	defer server.Close()
	source := modelDownloadTestSource(data, server.URL)
	part := filepath.Join(t.TempDir(), "model.gguf.part")
	if err := os.WriteFile(part, data[:prefix], 0o600); err != nil {
		t.Fatal(err)
	}
	if err := newTestDownloader(t, "4").fetch(source, part); err != nil {
		t.Fatal(err)
	}
	if _, err := verifyModelPullFile(part, source); err != nil {
		t.Fatal(err)
	}
}

func fmtSscanRange(header string, start *int64) (int, error) {
	value := strings.TrimPrefix(header, "bytes=")
	first, _, _ := strings.Cut(value, "-")
	return 1, json.Unmarshal([]byte(first), start)
}

func TestHuggingFaceTokenResolution(t *testing.T) {
	home := t.TempDir()
	t.Setenv("HF_TOKEN", "")
	t.Setenv("HF_TOKEN_PATH", "")
	t.Setenv("HF_HOME", "")
	t.Setenv("XDG_CACHE_HOME", home)
	if token, _ := huggingFaceToken(); token != "" {
		t.Fatalf("token without configuration: %q", token)
	}
	defaultPath := filepath.Join(home, "huggingface", "token")
	if err := writeHuggingFaceToken(defaultPath, "hf_from_default"); err != nil {
		t.Fatal(err)
	}
	if info, err := os.Stat(defaultPath); err != nil || info.Mode().Perm() != 0o600 {
		t.Fatalf("token file mode = %v, %v", info.Mode().Perm(), err)
	}
	if token, source := huggingFaceToken(); token != "hf_from_default" || source != defaultPath {
		t.Fatalf("default file: %q from %q", token, source)
	}
	custom := filepath.Join(t.TempDir(), "hf-home")
	t.Setenv("HF_HOME", custom)
	if err := os.MkdirAll(custom, 0o700); err != nil {
		t.Fatal(err)
	}
	os.WriteFile(filepath.Join(custom, "token"), []byte("hf_from_home\n"), 0o600)
	if token, _ := huggingFaceToken(); token != "hf_from_home" {
		t.Fatalf("HF_HOME token = %q", token)
	}
	explicit := filepath.Join(t.TempDir(), "explicit")
	os.WriteFile(explicit, []byte("hf_from_path"), 0o600)
	t.Setenv("HF_TOKEN_PATH", explicit)
	if token, _ := huggingFaceToken(); token != "hf_from_path" {
		t.Fatalf("HF_TOKEN_PATH token = %q", token)
	}
	t.Setenv("HF_TOKEN", "hf_from_env")
	if token, source := huggingFaceToken(); token != "hf_from_env" || source != "HF_TOKEN" {
		t.Fatalf("env token = %q from %q", token, source)
	}
	if env := withHuggingFaceToken([]string{"HF_TOKEN=deployment-specific"}); len(env) != 1 {
		t.Fatalf("existing HF_TOKEN overridden: %v", env)
	}
	if env := withHuggingFaceToken([]string{"PATH=/bin"}); env[len(env)-1] != "HF_TOKEN=hf_from_env" {
		t.Fatalf("server environment missing token: %v", env)
	}
}

// lmx auth hf --token-stdin validates with the Hub before writing, and never
// replaces a working token with a rejected one.
func TestAuthHFSavesOnlyValidatedTokens(t *testing.T) {
	hub := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if r.URL.Path != "/api/whoami-v2" || r.Header.Get("Authorization") != "Bearer hf_good" {
			w.WriteHeader(http.StatusUnauthorized)
			return
		}
		json.NewEncoder(w).Encode(map[string]any{"name": "lotto", "auth": map[string]any{"accessToken": map[string]any{"displayName": "laptop", "role": "read"}}})
	}))
	defer hub.Close()
	t.Setenv("HF_TOKEN", "")
	t.Setenv("HF_TOKEN_PATH", "")
	t.Setenv("HF_HOME", t.TempDir())
	args := cliArgs{opts: map[string]string{"hf-api-url": hub.URL}, flags: map[string]bool{"token-stdin": true, "quiet": true}}
	withStdin := func(input string) error {
		reader, writer, err := os.Pipe()
		if err != nil {
			t.Fatal(err)
		}
		writer.WriteString(input)
		writer.Close()
		saved := os.Stdin
		os.Stdin = reader
		defer func() { os.Stdin = saved }()
		return handleAuthHF("", args)
	}
	if err := withStdin("hf_good\n"); err != nil {
		t.Fatal(err)
	}
	if token, _ := huggingFaceToken(); token != "hf_good" {
		t.Fatalf("saved token = %q", token)
	}
	if err := withStdin("hf_revoked\n"); err == nil {
		t.Fatal("rejected token was accepted")
	}
	if token, _ := huggingFaceToken(); token != "hf_good" {
		t.Fatalf("rejected token replaced the working one: %q", token)
	}
	if err := handleAuthHF("logout", args); err != nil {
		t.Fatal(err)
	}
	if token, _ := huggingFaceToken(); token != "" {
		t.Fatalf("token survived logout: %q", token)
	}
}
