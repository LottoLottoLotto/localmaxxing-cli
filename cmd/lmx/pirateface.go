package main

import (
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"net"
	"net/http"
	"net/url"
	"os"
	"path/filepath"
	"strings"
	"time"

	alog "github.com/anacrolix/log"
	"github.com/anacrolix/torrent"
	"github.com/anacrolix/torrent/metainfo"
	"github.com/anacrolix/torrent/storage"
)

// Pirate Face (pirateface.co) lists Hugging Face models as BitTorrent swarms. With
// --source pirateface, model pull/snapshot fetch files from the swarm and check every
// file against Hugging Face's own hash, exactly as for a Hugging Face download; files
// the swarm does not supply in time come from Hugging Face.

const defaultPirateFaceURL = "https://pirateface.co"

// Variables so tests can shorten them.
var (
	// How long to wait for the torrent's metadata from peers.
	pirateFaceInfoTimeout = 60 * time.Second
	// A swarm that delivers no bytes for this long is abandoned for Hugging Face.
	pirateFaceStallTimeout = 30 * time.Second
)

type pirateFaceListing struct {
	RepoID   string `json:"repoId"`
	Revision string `json:"revision"`
	Seeders  int    `json:"seeders"`
	Leechers int    `json:"leechers"`
	Files    []struct {
		Path   string `json:"path"`
		SHA256 string `json:"sha256"`
	} `json:"files"`
	Infohash string `json:"infohash"`
	Magnet   string `json:"magnet"`
}

// modelDownloadSource reads --source: "huggingface" (default) or "pirateface".
func modelDownloadSource(args cliArgs) (string, error) {
	switch source := strings.ToLower(strings.TrimSpace(opt(args, "source"))); source {
	case "", "huggingface", "hf":
		return "huggingface", nil
	case "pirateface", "pirate-face", "pf":
		return "pirateface", nil
	default:
		return "", cliError{"invalid_option", "--source must be huggingface or pirateface", nil, nil}
	}
}

func pirateFaceURL(args cliArgs) string {
	if value := opt(args, "pirateface-url"); value != "" {
		return strings.TrimRight(value, "/")
	}
	if value := os.Getenv("LMX_PIRATEFACE_URL"); value != "" {
		return strings.TrimRight(value, "/")
	}
	return defaultPirateFaceURL
}

// lookupPirateFace returns the repository's listing, or nil when Pirate Face does not
// list it (or lists it only behind license terms the downloader must accept on the site).
func lookupPirateFace(client *http.Client, base, repo string) (*pirateFaceListing, error) {
	ctx, cancel := context.WithTimeout(context.Background(), 20*time.Second)
	defer cancel()
	req, err := http.NewRequestWithContext(ctx, http.MethodGet, base+"/api/torrents?"+url.Values{"repo": {repo}}.Encode(), nil)
	if err != nil {
		return nil, err
	}
	req.Header.Set("User-Agent", "localmaxxing-cli/"+version)
	response, err := client.Do(req)
	if err != nil {
		return nil, fmt.Errorf("Pirate Face lookup failed: %w", err)
	}
	defer response.Body.Close()
	if response.StatusCode == http.StatusNotFound {
		return nil, nil
	}
	if response.StatusCode != http.StatusOK {
		return nil, fmt.Errorf("Pirate Face lookup returned HTTP %d", response.StatusCode)
	}
	var listing pirateFaceListing
	if err := json.NewDecoder(io.LimitReader(response.Body, 8<<20)).Decode(&listing); err != nil {
		return nil, fmt.Errorf("invalid Pirate Face listing: %w", err)
	}
	if listing.Magnet == "" || !strings.HasPrefix(listing.Magnet, "magnet:?") {
		return nil, nil
	}
	if !modelPullHex(listing.Infohash, 20) {
		return nil, errors.New("Pirate Face listing has an invalid infohash")
	}
	if listing.Revision != "" && !modelPullHex(listing.Revision, 20) {
		return nil, errors.New("Pirate Face listing has an invalid revision")
	}
	listing.Infohash = strings.ToLower(listing.Infohash)
	listing.Revision = strings.ToLower(listing.Revision)
	return &listing, nil
}

// pirateFaceListingFor looks the repository up when --source pirateface is set. It
// returns the revision to download: the explicit --revision, else Pirate Face's pinned
// revision so the swarm's files match.
func pirateFaceListingFor(args cliArgs, client *http.Client, repo string) (*pirateFaceListing, string, error) {
	revision := opt(args, "revision")
	source, err := modelDownloadSource(args)
	if err != nil || source != "pirateface" {
		return nil, revision, err
	}
	listing, err := lookupPirateFace(client, pirateFaceURL(args), repo)
	if err != nil {
		return nil, revision, err
	}
	if listing == nil {
		printStatus(args, "pirateface_not_listed", map[string]any{"repo": repo, "fallback": "huggingface"})
		return nil, revision, nil
	}
	if revision == "" && listing.Revision != "" {
		revision = listing.Revision
	}
	printStatus(args, "pirateface_listing", map[string]any{"repo": repo, "revision": listing.Revision, "seeders": listing.Seeders, "infohash": listing.Infohash})
	return listing, revision, nil
}

// pirateFaceSwarm holds files the swarm delivered, waiting for acquireModelPull to
// verify them against Hugging Face's hashes and move them into the store.
type pirateFaceSwarm struct {
	dir   string
	files map[string]string // repository file -> path under dir
}

// take moves a swarm copy of source to part when its bytes match Hugging Face's
// hash. A copy that fails verification is deleted and the file comes from Hugging Face.
func (s *pirateFaceSwarm) take(args cliArgs, source modelPullSource, part string) (string, bool) {
	if s == nil {
		return "", false
	}
	path, ok := s.files[source.File]
	if !ok {
		return "", false
	}
	delete(s.files, source.File)
	digest, err := verifyModelPullFile(path, source)
	if err != nil {
		os.Remove(path)
		printStatus(args, "pirateface_file_rejected", map[string]any{"file": source.File, "reason": err.Error(), "fallback": "huggingface"})
		return "", false
	}
	os.Remove(part)
	os.Remove(part + ".chunks")
	if err := os.Rename(path, part); err != nil {
		return "", false
	}
	return digest, true
}

// cleanup removes the swarm's working directory once the command finished. After an
// interruption the directory stays so a rerun resumes from the pieces already on disk.
func (s *pirateFaceSwarm) cleanup(ctx context.Context) {
	if s == nil || ctx.Err() != nil {
		return
	}
	os.RemoveAll(s.dir)
	os.Remove(filepath.Dir(s.dir)) // the torrents directory, when no other download uses it
}

// pirateFaceWanted reports whether the swarm should fetch source: not already in the
// store or the Hugging Face cache, and listed by Pirate Face with matching contents.
func pirateFaceWanted(root string, listing *pirateFaceListing, source modelPullSource) bool {
	if destination, err := modelArtifactPath(root, source.Repo, source.Revision, source.File); err != nil {
		return false
	} else if _, err := os.Lstat(destination); err == nil {
		return false
	}
	for _, candidate := range modelPullCachePaths(source) {
		if info, err := os.Stat(candidate); err == nil && info.Size() == source.Size {
			return false
		}
	}
	for _, file := range listing.Files {
		if file.Path != source.File {
			continue
		}
		// Pirate Face records file SHA-256s; Hugging Face gives SHA-256 only for LFS files.
		return source.Integrity != "sha256" || strings.EqualFold(file.SHA256, source.ExpectedHash)
	}
	return false
}

// runPirateFaceSwarm downloads the wanted sources from the torrent. It returns the
// files it completed; anything else falls back to Hugging Face. It returns an error
// only when interrupted, keeping progress for a rerun.
func runPirateFaceSwarm(d *modelDownloader, root string, listing *pirateFaceListing, sources []modelPullSource) (*pirateFaceSwarm, error) {
	if listing == nil {
		return nil, nil
	}
	wanted := map[string]modelPullSource{}
	for _, source := range sources {
		if pirateFaceWanted(root, listing, source) {
			wanted[source.File] = source
		}
	}
	if len(wanted) == 0 {
		return nil, nil
	}
	swarm := &pirateFaceSwarm{dir: filepath.Join(root, "torrents", listing.Infohash), files: map[string]string{}}
	if err := os.MkdirAll(swarm.dir, 0o755); err != nil {
		return nil, err
	}
	completion, err := storage.NewBoltPieceCompletion(swarm.dir)
	if err != nil {
		return nil, fmt.Errorf("torrent progress database: %w", err)
	}
	config := torrent.NewDefaultClientConfig()
	config.DataDir = swarm.dir
	config.DefaultStorage = storage.NewFileOpts(storage.NewFileClientOpts{ClientBaseDir: swarm.dir, PieceCompletion: completion})
	config.ListenPort = 0
	config.Seed = false
	config.Logger = quietTorrentLogger()
	// Never ask the router to open ports (UPnP/NAT-PMP).
	config.NoDefaultPortForwarding = true
	client, err := torrent.NewClient(config)
	if err != nil {
		completion.Close()
		return nil, fmt.Errorf("start torrent client: %w", err)
	}
	defer func() {
		client.Close()
		<-client.Closed()
		completion.Close()
	}()
	t, err := client.AddMagnet(listing.Magnet)
	if err != nil {
		return nil, fmt.Errorf("Pirate Face magnet: %w", err)
	}
	// Peers named in the magnet itself (x.pe, BEP 9) are tried alongside trackers and DHT.
	if magnet, err := metainfo.ParseMagnetUri(listing.Magnet); err == nil {
		var peers []torrent.PeerInfo
		for _, hostPort := range magnet.Params["x.pe"] {
			if addr, err := net.ResolveTCPAddr("tcp", hostPort); err == nil {
				peers = append(peers, torrent.PeerInfo{Addr: addr, Source: torrent.PeerSourceDirect})
			}
		}
		t.AddPeers(peers)
	}
	printStatus(d.args, "pirateface_swarm_start", map[string]any{"files": len(wanted), "seeders": listing.Seeders})
	select {
	case <-t.GotInfo():
	case <-d.ctx.Done():
		return nil, d.ctx.Err()
	case <-time.After(pirateFaceInfoTimeout):
		printStatus(d.args, "pirateface_swarm_unavailable", map[string]any{"reason": "no peer sent the torrent metadata", "fallback": "huggingface"})
		return swarm, nil
	}

	type entry struct {
		file   *torrent.File
		source modelPullSource
	}
	var entries []entry
	var planned int64
	for _, file := range t.Files() {
		source, ok := wanted[file.DisplayPath()]
		if !ok || file.Length() != source.Size {
			continue
		}
		file.Download()
		entries = append(entries, entry{file, source})
		planned += source.Size
	}
	d.planned.Add(planned)
	completed := func() (done int64, finished int) {
		for _, e := range entries {
			n := e.file.BytesCompleted()
			done += n
			if n == e.file.Length() {
				finished++
			}
		}
		return done, finished
	}
	reported, _ := completed()
	d.received.Add(reported)
	lastProgress := time.Now()
	ticker := time.NewTicker(time.Second)
	defer ticker.Stop()
	for {
		done, finished := completed()
		if done > reported {
			d.received.Add(done - reported)
			reported = done
			lastProgress = time.Now()
		}
		if finished == len(entries) {
			break
		}
		if time.Since(lastProgress) > pirateFaceStallTimeout {
			printStatus(d.args, "pirateface_swarm_stalled", map[string]any{"filesDone": finished, "files": len(entries), "fallback": "huggingface"})
			break
		}
		select {
		case <-d.ctx.Done():
			return nil, d.ctx.Err()
		case <-ticker.C:
		}
	}
	var fromSwarm int
	for _, e := range entries {
		path := filepath.Join(swarm.dir, filepath.FromSlash(e.file.Path()))
		if e.file.BytesCompleted() == e.file.Length() {
			swarm.files[e.source.File] = path
			fromSwarm++
		} else {
			// Hugging Face supplies the rest; don't keep a second partial copy on disk.
			d.planned.Add(-(e.file.Length() - e.file.BytesCompleted()))
		}
	}
	printStatus(d.args, "pirateface_swarm_done", map[string]any{"filesFromSwarm": fromSwarm, "filesFromHuggingFace": len(wanted) - fromSwarm})
	return swarm, nil
}

// pirateFacePlan summarizes the listing for dry-run plans; nil when not using Pirate Face.
func pirateFacePlan(listing *pirateFaceListing) map[string]any {
	if listing == nil {
		return nil
	}
	return map[string]any{"revision": listing.Revision, "seeders": listing.Seeders, "infohash": listing.Infohash, "files": len(listing.Files)}
}

// quietTorrentLogger discards the torrent library's logging; lmx reports progress itself.
func quietTorrentLogger() alog.Logger {
	logger := alog.Default
	logger.SetHandlers(alog.DiscardHandler)
	return logger
}
