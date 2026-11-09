package main

import (
	"crypto/sha256"
	"fmt"
	"io"
	"net"
	"os"
	"path/filepath"
	"strings"
	"testing"
	"time"

	"github.com/anacrolix/torrent"
)

func freeTCPPort(t *testing.T) int {
	t.Helper()
	listener, err := net.Listen("tcp", "127.0.0.1:0")
	if err != nil {
		t.Fatal(err)
	}
	defer listener.Close()
	return listener.Addr().(*net.TCPAddr).Port
}

func fileDigest(t *testing.T, path string) [32]byte {
	t.Helper()
	data, err := os.ReadFile(path)
	if err != nil {
		t.Fatal(err)
	}
	return sha256.Sum256(data)
}

// A snapshot downloaded through Pirate Face is seeded from the store to a new peer after
// the original seeder left, without modifying the store, and seeding stops at --seed-ratio.
func TestModelSeedServesSnapshotFromStoreUntilRatio(t *testing.T) {
	t.Setenv("XDG_DATA_HOME", t.TempDir())
	upstream := map[string]string{
		"config.json":       `{"model_type":"test"}`,
		"model.safetensors": strings.Repeat("weights!", 9000),
	}
	magnet, infohash, original := seedPirateFaceTorrent(t, upstream)
	var listed []map[string]string
	for name, data := range upstream {
		listed = append(listed, map[string]string{"path": name, "sha256": modelPullTestSource(data).ExpectedHash})
	}
	hfDownloads := map[string]int{}
	hf, pf := pirateFaceTestServers(t, upstream, map[string]any{"repoId": "owner/model", "seeders": 1, "files": listed, "infohash": infohash, "magnet": magnet}, hfDownloads)
	root := filepath.Join(t.TempDir(), "store")
	args := pirateFaceTestArgs(t, hf, pf, root)
	if err := handleModelSnapshot("owner/model", args); err != nil {
		t.Fatal(err)
	}
	snapshot := modelSnapshotTestResult(t, args)
	before := map[string][32]byte{}
	for _, file := range snapshot.Files {
		before[file.File] = fileDigest(t, file.Artifact.Path)
	}
	// Only lmx has the files now; the metadata comes from lmx's cache, not from a peer.
	original.Close()

	port := freeTCPPort(t)
	seedArgs := pirateFaceTestArgs(t, hf, pf, root)
	seedArgs.opts["listen-port"] = fmt.Sprint(port)
	seedArgs.opts["seed-ratio"] = "0.5"
	seedArgs.flags["no-dht"] = true
	done := make(chan error, 1)
	go func() { done <- handleModelSeed(snapshot.ID, seedArgs) }()

	config := torrent.NewDefaultClientConfig()
	config.DataDir = t.TempDir()
	config.ListenPort = 0
	config.NoDHT = true
	config.NoDefaultPortForwarding = true
	config.Logger = quietTorrentLogger()
	leecher, err := torrent.NewClient(config)
	if err != nil {
		t.Fatal(err)
	}
	defer leecher.Close()
	leech, err := leecher.AddMagnet(strings.SplitN(magnet, "&x.pe=", 2)[0] + fmt.Sprintf("&x.pe=127.0.0.1:%d", port))
	if err != nil {
		t.Fatal(err)
	}
	addMagnetPeers(leech, fmt.Sprintf("magnet:?xt=urn:btih:%s&x.pe=127.0.0.1:%d", infohash, port))
	deadline := time.After(60 * time.Second)
	for {
		select {
		case <-leech.GotInfo():
		case <-deadline:
			t.Fatal("leecher never got the torrent metadata from lmx")
		case <-time.After(200 * time.Millisecond):
			addMagnetPeers(leech, fmt.Sprintf("magnet:?xt=urn:btih:%s&x.pe=127.0.0.1:%d", infohash, port))
			continue
		}
		break
	}
	leech.DownloadAll()
	for leech.BytesMissing() > 0 {
		select {
		case <-deadline:
			t.Fatalf("leecher still missing %d bytes", leech.BytesMissing())
		case <-time.After(200 * time.Millisecond):
		}
	}
	for _, file := range leech.Files() {
		reader := file.NewReader()
		data, err := io.ReadAll(io.LimitReader(reader, file.Length()))
		reader.Close()
		if err != nil || string(data) != upstream[file.DisplayPath()] {
			t.Fatalf("leecher %s: %d bytes, want %d (%q), %v", file.DisplayPath(), len(data), len(upstream[file.DisplayPath()]), file.DisplayPath(), err)
		}
	}
	select {
	case err := <-done:
		if err != nil {
			t.Fatal(err)
		}
	case <-time.After(30 * time.Second):
		t.Fatal("seeding did not stop at --seed-ratio")
	}
	for _, file := range snapshot.Files {
		if fileDigest(t, file.Artifact.Path) != before[file.File] {
			t.Fatalf("seeding modified %s in the store", file.File)
		}
	}
}

func TestModelSeedSkipsSnapshotsPirateFaceListsAtAnotherRevision(t *testing.T) {
	t.Setenv("XDG_DATA_HOME", t.TempDir())
	upstream := map[string]string{"model.safetensors": "weights"}
	listing := map[string]any{"repoId": "owner/model", "seeders": 1, "infohash": strings.Repeat("cd", 20), "magnet": "magnet:?xt=urn:btih:" + strings.Repeat("cd", 20)}
	hf, pf := pirateFaceTestServers(t, upstream, nil, map[string]int{})
	root := filepath.Join(t.TempDir(), "store")
	args := modelPullTestArgs(t, hf.URL, root)
	if err := handleModelSnapshot("owner/model", args); err != nil {
		t.Fatal(err)
	}
	// Pirate Face now lists the repository, but at a different commit.
	listing["revision"] = strings.Repeat("b", 40)
	_, pf = pirateFaceTestServers(t, upstream, listing, map[string]int{})
	seedArgs := pirateFaceTestArgs(t, hf, pf, root)
	err := handleModelSeed("owner/model", seedArgs)
	if cli, ok := err.(cliError); !ok || cli.Code != "nothing_to_seed" {
		t.Fatalf("seed error = %v", err)
	}
}

func TestParseByteRate(t *testing.T) {
	for value, want := range map[string]int64{"5MB": 5_000_000, "500kb": 500_000, "1MiB": 1 << 20, "1.5 GiB": 3 << 29, "2000": 2000, "10MB/s": 10_000_000} {
		if got, err := parseByteRate(value); err != nil || got != want {
			t.Fatalf("parseByteRate(%q) = %d, %v; want %d", value, got, err, want)
		}
	}
	for _, value := range []string{"", "fast", "-1MB", "0", "0.1B"} {
		if _, err := parseByteRate(value); err == nil {
			t.Fatalf("parseByteRate(%q) accepted", value)
		}
	}
}
