package main

import (
	"encoding/json"
	"fmt"
	"net/http"
	"net/http/httptest"
	"net/url"
	"os"
	"path/filepath"
	"strings"
	"testing"
	"time"

	"github.com/anacrolix/torrent"
	"github.com/anacrolix/torrent/bencode"
	"github.com/anacrolix/torrent/metainfo"
)

// seedPirateFaceTorrent seeds files from a local client and returns a magnet that
// names the seeder as a peer (x.pe), so no tracker or DHT is needed.
func seedPirateFaceTorrent(t *testing.T, files map[string]string) (magnet, infohash string, seeder *torrent.Client) {
	t.Helper()
	dir := t.TempDir()
	content := filepath.Join(dir, "model")
	for name, data := range files {
		path := filepath.Join(content, filepath.FromSlash(name))
		if err := os.MkdirAll(filepath.Dir(path), 0o755); err != nil {
			t.Fatal(err)
		}
		if err := os.WriteFile(path, []byte(data), 0o644); err != nil {
			t.Fatal(err)
		}
	}
	info := metainfo.Info{PieceLength: 16 << 10}
	if err := info.BuildFromFilePath(content); err != nil {
		t.Fatal(err)
	}
	infoBytes, err := bencode.Marshal(info)
	if err != nil {
		t.Fatal(err)
	}
	mi := metainfo.MetaInfo{InfoBytes: infoBytes}
	config := torrent.NewDefaultClientConfig()
	config.DataDir = dir
	config.Seed = true
	config.ListenHost = func(string) string { return "127.0.0.1" }
	config.ListenPort = 0
	config.NoDHT = true
	config.DisableUTP = true
	config.DisableIPv6 = true
	config.Logger = quietTorrentLogger()
	config.NoDefaultPortForwarding = true
	seeder, err = torrent.NewClient(config)
	if err != nil {
		t.Fatal(err)
	}
	t.Cleanup(func() { seeder.Close() })
	seeded, err := seeder.AddTorrent(&mi)
	if err != nil {
		t.Fatal(err)
	}
	<-seeded.GotInfo()
	seeded.VerifyData()
	hash := mi.HashInfoBytes()
	link := mi.Magnet(&hash, &info)
	link.Params = url.Values{"x.pe": {fmt.Sprintf("127.0.0.1:%d", seeder.LocalPort())}}
	return link.String(), hash.HexString(), seeder
}

// pirateFaceTestServers starts a Hugging Face server (counting file downloads) and a
// Pirate Face server returning listing for owner/model (404 when nil).
func pirateFaceTestServers(t *testing.T, upstream map[string]string, listing map[string]any, hfDownloads map[string]int) (hf, pf *httptest.Server) {
	t.Helper()
	hf = modelSnapshotTestServer(t, upstream, func(w http.ResponseWriter, _ *http.Request, name, data string) {
		hfDownloads[name]++
		fmt.Fprint(w, data)
	})
	t.Cleanup(hf.Close)
	pf = httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if r.URL.Path != "/api/torrents" || listing == nil || r.URL.Query().Get("repo") != "owner/model" {
			http.NotFound(w, r)
			return
		}
		json.NewEncoder(w).Encode(listing)
	}))
	t.Cleanup(pf.Close)
	return hf, pf
}

func pirateFaceTestArgs(t *testing.T, hf, pf *httptest.Server, root string) cliArgs {
	args := modelPullTestArgs(t, hf.URL, root)
	args.opts["source"] = "pirateface"
	args.opts["pirateface-url"] = pf.URL
	args.flags["quiet"] = true
	return args
}

func TestModelSnapshotFromPirateFaceSwarmVerifiesAgainstHuggingFace(t *testing.T) {
	upstream := map[string]string{
		"config.json":       `{"model_type":"test"}`,
		"model.safetensors": strings.Repeat("weights!", 9000), // spans several pieces
		"tokenizer.json":    `{"vocab":{}}`,
	}
	// The torrent's tokenizer.json differs from Hugging Face's: it must be rejected and
	// fetched from Hugging Face; a file missing from the torrent also comes from there.
	torrentFiles := map[string]string{
		"config.json":       upstream["config.json"],
		"model.safetensors": upstream["model.safetensors"],
		"tokenizer.json":    `{"vocab":{"x"}}`,
	}
	upstream["README.md"] = "readme"
	magnet, infohash, _ := seedPirateFaceTorrent(t, torrentFiles)
	var listed []map[string]string
	for name := range torrentFiles {
		listed = append(listed, map[string]string{"path": name, "sha256": modelPullTestSource(torrentFiles[name]).ExpectedHash})
	}
	hfDownloads := map[string]int{}
	hf, pf := pirateFaceTestServers(t, upstream, map[string]any{"repoId": "owner/model", "seeders": 1, "files": listed, "infohash": infohash, "magnet": magnet}, hfDownloads)
	root := filepath.Join(t.TempDir(), "store")
	args := pirateFaceTestArgs(t, hf, pf, root)

	if err := handleModelSnapshot("owner/model", args); err != nil {
		t.Fatal(err)
	}
	snapshot := modelSnapshotTestResult(t, args)
	if snapshot.State != "ready" {
		t.Fatalf("snapshot state = %q", snapshot.State)
	}
	for name, expected := range upstream {
		data, err := os.ReadFile(filepath.Join(snapshot.Path, filepath.FromSlash(name)))
		if err != nil || string(data) != expected {
			t.Fatalf("%s = %q, %v", name, data, err)
		}
	}
	// The listing's tokenizer.json SHA-256 does not match Hugging Face's, so it is never
	// taken from the swarm; README.md is not in the torrent.
	want := map[string]int{"tokenizer.json": 1, "README.md": 1}
	if fmt.Sprint(hfDownloads) != fmt.Sprint(want) {
		t.Fatalf("Hugging Face downloads = %v, want %v", hfDownloads, want)
	}
	if _, err := os.Stat(filepath.Join(root, "torrents")); !os.IsNotExist(err) {
		t.Fatalf("torrent working directory left behind: %v", err)
	}
}

func TestModelSnapshotRejectsSwarmBytesThatFailHuggingFaceHash(t *testing.T) {
	upstream := map[string]string{"model.safetensors": strings.Repeat("good", 5000)}
	// The listing claims Hugging Face's SHA-256, but the swarm serves other bytes.
	bad := map[string]string{"model.safetensors": strings.Repeat("evil", 5000)}
	magnet, infohash, _ := seedPirateFaceTorrent(t, bad)
	listed := []map[string]string{{"path": "model.safetensors", "sha256": modelPullTestSource(upstream["model.safetensors"]).ExpectedHash}}
	hfDownloads := map[string]int{}
	hf, pf := pirateFaceTestServers(t, upstream, map[string]any{"repoId": "owner/model", "seeders": 1, "files": listed, "infohash": infohash, "magnet": magnet}, hfDownloads)
	root := filepath.Join(t.TempDir(), "store")
	args := pirateFaceTestArgs(t, hf, pf, root)

	if err := handleModelSnapshot("owner/model", args); err != nil {
		t.Fatal(err)
	}
	snapshot := modelSnapshotTestResult(t, args)
	data, err := os.ReadFile(filepath.Join(snapshot.Path, "model.safetensors"))
	if err != nil || string(data) != upstream["model.safetensors"] {
		t.Fatalf("served bytes = %.20q…, %v", data, err)
	}
	if hfDownloads["model.safetensors"] != 1 {
		t.Fatalf("Hugging Face downloads = %v", hfDownloads)
	}
}

func TestModelSnapshotFallsBackWhenPirateFaceCannotHelp(t *testing.T) {
	previous := pirateFaceInfoTimeout
	pirateFaceInfoTimeout = 2 * time.Second
	t.Cleanup(func() { pirateFaceInfoTimeout = previous })
	upstream := map[string]string{"model.safetensors": "weights"}
	listed := []map[string]string{{"path": "model.safetensors", "sha256": modelPullTestSource(upstream["model.safetensors"]).ExpectedHash}}
	// No peer ever answers this magnet.
	deadMagnet := "magnet:?xt=urn:btih:" + strings.Repeat("ab", 20) + "&x.pe=127.0.0.1:9"
	for name, listing := range map[string]map[string]any{
		"not listed": nil,
		"no peers":   {"repoId": "owner/model", "seeders": 0, "files": listed, "infohash": strings.Repeat("ab", 20), "magnet": deadMagnet},
	} {
		t.Run(name, func(t *testing.T) {
			hfDownloads := map[string]int{}
			hf, pf := pirateFaceTestServers(t, upstream, listing, hfDownloads)
			args := pirateFaceTestArgs(t, hf, pf, filepath.Join(t.TempDir(), "store"))
			if err := handleModelSnapshot("owner/model", args); err != nil {
				t.Fatal(err)
			}
			if snapshot := modelSnapshotTestResult(t, args); snapshot.State != "ready" || hfDownloads["model.safetensors"] != 1 {
				t.Fatalf("state %q, Hugging Face downloads %v", snapshot.State, hfDownloads)
			}
		})
	}
}

func TestModelDownloadSourceOption(t *testing.T) {
	for value, want := range map[string]string{"": "huggingface", "hf": "huggingface", "PirateFace": "pirateface", "pf": "pirateface"} {
		got, err := modelDownloadSource(cliArgs{opts: map[string]string{"source": value}})
		if err != nil || got != want {
			t.Fatalf("--source %q = %q, %v", value, got, err)
		}
	}
	if _, err := modelDownloadSource(cliArgs{opts: map[string]string{"source": "ipfs"}}); err == nil {
		t.Fatal("unknown source accepted")
	}
}
