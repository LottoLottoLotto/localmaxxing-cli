package main

import (
	"crypto/sha1"
	"crypto/sha256"
	"encoding/hex"
	"encoding/json"
	"fmt"
	"net/http"
	"net/http/httptest"
	"os"
	"path/filepath"
	"strings"
	"testing"
)

func modelPullTestSource(data string) modelPullSource {
	digest := sha256.Sum256([]byte(data))
	return modelPullSource{Repo: "owner/model", Revision: strings.Repeat("a", 40), File: "weights/model.gguf", Size: int64(len(data)), Integrity: "sha256", ExpectedHash: hex.EncodeToString(digest[:])}
}

func TestModelPullRangeResponses(t *testing.T) {
	const data = "complete model bytes"
	for _, mode := range []string{"ignored", "wrong-start", "wrong-total", "malformed"} {
		t.Run(mode, func(t *testing.T) {
			source := modelPullTestSource(data)
			prefix := data[:5]
			part := filepath.Join(t.TempDir(), "model.part")
			if err := os.WriteFile(part, []byte(prefix), 0o600); err != nil {
				t.Fatal(err)
			}
			server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
				if r.Header.Get("Range") != "bytes=5-" {
					t.Errorf("resume range = %q", r.Header.Get("Range"))
				}
				if mode == "ignored" {
					fmt.Fprint(w, data)
					return
				}
				switch mode {
				case "wrong-start":
					w.Header().Set("Content-Range", fmt.Sprintf("bytes 4-%d/%d", len(data)-1, len(data)))
				case "wrong-total":
					w.Header().Set("Content-Range", fmt.Sprintf("bytes 5-%d/%d", len(data)-1, len(data)+1))
				case "malformed":
					w.Header().Set("Content-Range", "bytes five-last/*")
				}
				w.WriteHeader(http.StatusPartialContent)
				fmt.Fprint(w, data[5:])
			}))
			defer server.Close()
			source.URL = server.URL
			client := modelPullHTTPClient()
			defer client.CloseIdleConnections()
			err := downloadModelPull(client, source, part)
			got, readErr := os.ReadFile(part)
			if readErr != nil {
				t.Fatal(readErr)
			}
			if mode == "ignored" {
				if err != nil {
					t.Fatal(err)
				}
				if string(got) != data {
					t.Fatalf("ignored range appended full body: %q", got)
				}
				if _, err := verifyModelPullFile(part, source); err != nil {
					t.Fatal(err)
				}
			} else {
				if err == nil {
					t.Fatal("invalid range was accepted")
				}
				if string(got) != prefix {
					t.Fatalf("invalid range damaged resume prefix: %q", got)
				}
			}
		})
	}
}

func TestModelPullInterruptedDownloadResumes(t *testing.T) {
	const data = "a model too large for one interrupted response"
	source := modelPullTestSource(data)
	requests := 0
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		requests++
		if requests == 1 {
			w.Header().Set("Content-Length", fmt.Sprint(len(data)))
			fmt.Fprint(w, data[:7])
			return
		}
		if r.Header.Get("Range") != "bytes=7-" {
			t.Errorf("resume range = %q", r.Header.Get("Range"))
		}
		w.Header().Set("Content-Range", fmt.Sprintf("bytes 7-%d/%d", len(data)-1, len(data)))
		w.WriteHeader(http.StatusPartialContent)
		fmt.Fprint(w, data[7:])
	}))
	defer server.Close()
	source.URL = server.URL
	part := filepath.Join(t.TempDir(), "model.part")
	client := modelPullHTTPClient()
	defer client.CloseIdleConnections()
	if err := downloadModelPull(client, source, part); err == nil {
		t.Fatal("interrupted response accepted")
	}
	prefix, err := os.ReadFile(part)
	if err != nil || string(prefix) != data[:7] {
		t.Fatalf("partial bytes = %q, error = %v", prefix, err)
	}
	if err := downloadModelPull(client, source, part); err != nil {
		t.Fatal(err)
	}
	if _, err := verifyModelPullFile(part, source); err != nil {
		t.Fatal(err)
	}
}

func modelPullTestServer(t *testing.T, source modelPullSource, download http.HandlerFunc) *httptest.Server {
	t.Helper()
	return httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if r.URL.Path == "/api/models/owner/model/revision/main" {
			if r.URL.Query().Get("blobs") != "true" {
				t.Error("metadata request omitted blob metadata")
			}
			json.NewEncoder(w).Encode(map[string]any{"sha": source.Revision, "siblings": []any{map[string]any{"rfilename": source.File, "size": source.Size, "lfs": map[string]any{"sha256": source.ExpectedHash, "size": source.Size}}}})
			return
		}
		if r.URL.Path != "/owner/model/resolve/"+source.Revision+"/"+source.File {
			t.Errorf("request was not pinned to resolved commit: %s", r.URL.Path)
			http.NotFound(w, r)
			return
		}
		download(w, r)
	}))
}

func modelPullTestArgs(t *testing.T, serverURL, root string) cliArgs {
	t.Helper()
	t.Setenv("HF_HUB_CACHE", filepath.Join(t.TempDir(), "empty-cache"))
	return cliArgs{opts: map[string]string{"hf-api-url": serverURL, "models-dir": root, "file": "weights/model.gguf", "out": filepath.Join(t.TempDir(), "result.json")}, flags: map[string]bool{"yes": true}}
}

func TestModelPullRejectsHashMismatchBeforePromotion(t *testing.T) {
	const data = "the correct weights"
	source := modelPullTestSource(data)
	server := modelPullTestServer(t, source, func(w http.ResponseWriter, r *http.Request) { fmt.Fprint(w, strings.Repeat("x", len(data))) })
	defer server.Close()
	root := filepath.Join(t.TempDir(), "store")
	args := modelPullTestArgs(t, server.URL, root)
	if err := handleModelPull(source.Repo, args); err == nil || !strings.Contains(err.Error(), "integrity mismatch") {
		t.Fatalf("hash mismatch error = %v", err)
	}
	destination, err := modelArtifactPath(root, source.Repo, source.Revision, source.File)
	if err != nil {
		t.Fatal(err)
	}
	for _, path := range []string{destination, destination + ".part"} {
		if _, err := os.Stat(path); !os.IsNotExist(err) {
			t.Fatalf("corrupt file survived at %s: %v", path, err)
		}
	}
}

func TestModelPullPinsAndReusesVerifiedDownload(t *testing.T) {
	const data = "verified model data"
	source := modelPullTestSource(data)
	downloads := 0
	server := modelPullTestServer(t, source, func(w http.ResponseWriter, r *http.Request) { downloads++; fmt.Fprint(w, data) })
	defer server.Close()
	root := filepath.Join(t.TempDir(), "store")
	args := modelPullTestArgs(t, server.URL, root)
	for range 2 {
		if err := handleModelPull(source.Repo, args); err != nil {
			t.Fatal(err)
		}
	}
	if downloads != 1 {
		t.Fatalf("repeat pull downloaded %d times", downloads)
	}
	result, err := os.ReadFile(args.opts["out"])
	if err != nil {
		t.Fatal(err)
	}
	var output struct {
		Artifact modelArtifact `json:"artifact"`
	}
	if err := json.Unmarshal(result, &output); err != nil {
		t.Fatal(err)
	}
	if !output.Artifact.UpstreamVerified || output.Artifact.SHA256 != source.ExpectedHash {
		t.Fatalf("verified artifact = %+v", output.Artifact)
	}
	contents, err := os.ReadFile(output.Artifact.Path)
	if err != nil || string(contents) != data {
		t.Fatalf("artifact bytes = %q, error = %v", contents, err)
	}
}

func TestModelPullDryRunOverridesConsent(t *testing.T) {
	source := modelPullTestSource("weights")
	server := modelPullTestServer(t, source, func(w http.ResponseWriter, r *http.Request) { t.Error("dry run downloaded bytes") })
	defer server.Close()
	root := filepath.Join(t.TempDir(), "store")
	args := modelPullTestArgs(t, server.URL, root)
	args.flags["dry-run"] = true
	if err := handleModelPull(source.Repo, args); err != nil {
		t.Fatal(err)
	}
	if _, err := os.Stat(root); !os.IsNotExist(err) {
		t.Fatalf("dry run mutated model storage: %v", err)
	}
}

func TestModelPullDefaultsToPlanWithoutConsent(t *testing.T) {
	source := modelPullTestSource("weights")
	server := modelPullTestServer(t, source, func(w http.ResponseWriter, r *http.Request) { t.Error("plan downloaded bytes without consent") })
	defer server.Close()
	root := filepath.Join(t.TempDir(), "store")
	args := modelPullTestArgs(t, server.URL, root)
	delete(args.flags, "yes")
	if err := handleModelPull(source.Repo, args); err != nil {
		t.Fatal(err)
	}
	if _, err := os.Stat(root); !os.IsNotExist(err) {
		t.Fatalf("plan created managed storage: %v", err)
	}
	result, err := os.ReadFile(args.opts["out"])
	if err != nil {
		t.Fatal(err)
	}
	var output struct {
		Source           modelPullSource `json:"source"`
		UpstreamVerified bool            `json:"upstreamVerified"`
	}
	if err := json.Unmarshal(result, &output); err != nil {
		t.Fatal(err)
	}
	if output.Source.Revision != source.Revision || output.Source.ExpectedHash != source.ExpectedHash || output.UpstreamVerified {
		t.Fatalf("plan failed to pin expected integrity without claiming verification: %+v", output)
	}
}

func TestModelPullLocalDigestDoesNotClaimUpstreamVerification(t *testing.T) {
	const data = "weights without upstream digest"
	source := modelPullTestSource(data)
	source.ExpectedHash = ""
	server := modelPullTestServer(t, source, func(w http.ResponseWriter, r *http.Request) { fmt.Fprint(w, data) })
	defer server.Close()
	args := modelPullTestArgs(t, server.URL, filepath.Join(t.TempDir(), "store"))
	if err := handleModelPull(source.Repo, args); err != nil {
		t.Fatal(err)
	}
	result, err := os.ReadFile(args.opts["out"])
	if err != nil {
		t.Fatal(err)
	}
	var output struct {
		Artifact modelArtifact `json:"artifact"`
	}
	if err := json.Unmarshal(result, &output); err != nil {
		t.Fatal(err)
	}
	expected := sha256.Sum256([]byte(data))
	if output.Artifact.UpstreamVerified || output.Artifact.Integrity != "local-sha256" || output.Artifact.SHA256 != hex.EncodeToString(expected[:]) {
		t.Fatalf("local digest misrepresented: %+v", output.Artifact)
	}
}

func TestModelPullCacheReferenceDoesNotCopyOrModifySource(t *testing.T) {
	const data = "cached model bytes"
	source := modelPullTestSource(data)
	server := modelPullTestServer(t, source, func(w http.ResponseWriter, r *http.Request) { t.Error("verified cache caused a download") })
	defer server.Close()
	root := filepath.Join(t.TempDir(), "store")
	args := modelPullTestArgs(t, server.URL, root)
	cache := filepath.Join(t.TempDir(), "cache")
	t.Setenv("HF_HUB_CACHE", cache)
	blob := filepath.Join(cache, "models--owner--model", "blobs", source.ExpectedHash)
	if err := os.MkdirAll(filepath.Dir(blob), 0o755); err != nil {
		t.Fatal(err)
	}
	if err := os.WriteFile(blob, []byte(data), 0o600); err != nil {
		t.Fatal(err)
	}
	if err := handleModelPull(source.Repo, args); err != nil {
		t.Fatal(err)
	}
	result, err := os.ReadFile(args.opts["out"])
	if err != nil {
		t.Fatal(err)
	}
	var output struct {
		Artifact modelArtifact `json:"artifact"`
	}
	if err := json.Unmarshal(result, &output); err != nil {
		t.Fatal(err)
	}
	if output.Artifact.Managed || output.Artifact.Path != blob || !output.Artifact.UpstreamVerified {
		t.Fatalf("cache registration = %+v", output.Artifact)
	}
	contents, err := os.ReadFile(blob)
	if err != nil || string(contents) != data {
		t.Fatalf("cache source bytes = %q, error = %v", contents, err)
	}
}

func TestModelPullDoesNotForwardTokenAcrossOrigins(t *testing.T) {
	t.Setenv("HF_TOKEN", "secret-test-token")
	target := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if r.Header.Get("Authorization") != "" {
			t.Error("token leaked to redirect origin")
		}
		fmt.Fprint(w, "bytes")
	}))
	defer target.Close()
	origin := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if r.Header.Get("Authorization") != "Bearer secret-test-token" {
			t.Error("HF request missing token")
		}
		http.Redirect(w, r, target.URL, http.StatusFound)
	}))
	defer origin.Close()
	source := modelPullTestSource("bytes")
	source.URL = origin.URL
	client := modelPullHTTPClient()
	defer client.CloseIdleConnections()
	if err := downloadModelPull(client, source, filepath.Join(t.TempDir(), "model.part")); err != nil {
		t.Fatal(err)
	}
}

func TestModelPullGitBlobIntegrity(t *testing.T) {
	const data = "plain git blob bytes"
	digest := sha1.Sum([]byte(fmt.Sprintf("blob %d\x00%s", len(data), data)))
	source := modelPullTestSource(data)
	source.Integrity = "git-blob-sha1"
	source.ExpectedHash = hex.EncodeToString(digest[:])
	path := filepath.Join(t.TempDir(), "model")
	if err := os.WriteFile(path, []byte(data), 0o600); err != nil {
		t.Fatal(err)
	}
	if _, err := verifyModelPullFile(path, source); err != nil {
		t.Fatal(err)
	}
	if err := os.WriteFile(path, []byte(strings.Repeat("x", len(data))), 0o600); err != nil {
		t.Fatal(err)
	}
	if _, err := verifyModelPullFile(path, source); err == nil {
		t.Fatal("same-size Git blob corruption accepted")
	}
}
