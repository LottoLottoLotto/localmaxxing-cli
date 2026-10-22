package main

import (
	"errors"
	"fmt"
	"math"
	"net"
	"os"
	"path/filepath"
	"strconv"
	"strings"
	"time"

	alog "github.com/anacrolix/log"
	"github.com/anacrolix/torrent"
	"github.com/anacrolix/torrent/metainfo"
	"github.com/anacrolix/torrent/storage"
	"golang.org/x/time/rate"
)

// torrentOptions are the BitTorrent settings shared by --source pirateface downloads
// and lmx model seed. Nothing is uploaded unless asked for.
type torrentOptions struct {
	// Upload shares pieces with other peers while downloading (--upload).
	Upload        bool
	UploadLimit   int64 // bytes/s; 0 = unlimited
	DownloadLimit int64 // bytes/s; 0 = unlimited
	ListenPort    int   // 0 = random
	NoDHT         bool
	// SeedRatio and SeedTime bound seeding: after a download they turn seeding on, and
	// for lmx model seed they end it. Zero means no limit of that kind.
	SeedRatio float64
	SeedTime  time.Duration
}

func parseTorrentOptions(args cliArgs) (torrentOptions, error) {
	options := torrentOptions{Upload: hasFlag(args, "upload"), NoDHT: hasFlag(args, "no-dht")}
	var err error
	if options.UploadLimit, err = byteRateOption(args, "upload-limit"); err != nil {
		return options, err
	}
	if options.DownloadLimit, err = byteRateOption(args, "download-limit"); err != nil {
		return options, err
	}
	if value := opt(args, "listen-port"); value != "" {
		port, err := strconv.Atoi(value)
		if err != nil || port < 1 || port > 65535 {
			return options, cliError{"invalid_option", "--listen-port must be a port number between 1 and 65535", nil, nil}
		}
		options.ListenPort = port
	}
	if value := opt(args, "seed-ratio"); value != "" {
		ratio, err := strconv.ParseFloat(value, 64)
		if err != nil || ratio <= 0 || math.IsInf(ratio, 0) || math.IsNaN(ratio) {
			return options, cliError{"invalid_option", "--seed-ratio must be a positive number, e.g. 1.0 (upload as much as the model's size)", nil, nil}
		}
		options.SeedRatio = ratio
	}
	if value := opt(args, "seed-time"); value != "" {
		duration, err := time.ParseDuration(value)
		if err != nil || duration <= 0 {
			return options, cliError{"invalid_option", "--seed-time must be a positive duration, e.g. 30m or 2h", nil, nil}
		}
		options.SeedTime = duration
	}
	return options, nil
}

// seedAfterDownload reports whether a pull/snapshot should keep seeding once done.
func (o torrentOptions) seedAfterDownload() bool { return o.SeedRatio > 0 || o.SeedTime > 0 }

// byteRateOption parses a rate such as 5MB, 500KB, 1.5MiB or 2000000 (bytes per second).
func byteRateOption(args cliArgs, key string) (int64, error) {
	value := strings.TrimSpace(opt(args, key))
	if value == "" {
		return 0, nil
	}
	rateValue, err := parseByteRate(value)
	if err != nil {
		return 0, cliError{"invalid_option", fmt.Sprintf("--%s must be a rate in bytes per second such as 5MB, 500KB or 1MiB", key), nil, nil}
	}
	return rateValue, nil
}

func parseByteRate(value string) (int64, error) {
	upper := strings.ToUpper(strings.TrimSuffix(strings.TrimSuffix(strings.TrimSpace(value), "/s"), "/S"))
	units := []struct {
		suffix     string
		multiplier float64
	}{{"KIB", 1 << 10}, {"MIB", 1 << 20}, {"GIB", 1 << 30}, {"KB", 1e3}, {"MB", 1e6}, {"GB", 1e9}, {"K", 1e3}, {"M", 1e6}, {"G", 1e9}, {"B", 1}}
	multiplier := 1.0
	for _, unit := range units {
		if strings.HasSuffix(upper, unit.suffix) {
			upper, multiplier = strings.TrimSpace(strings.TrimSuffix(upper, unit.suffix)), unit.multiplier
			break
		}
	}
	number, err := strconv.ParseFloat(upper, 64)
	if err != nil || number <= 0 || math.IsInf(number, 0) || math.IsNaN(number) {
		return 0, errors.New("invalid rate")
	}
	bytes := number * multiplier
	if bytes < 1 || bytes > float64(1<<50) {
		return 0, errors.New("rate out of range")
	}
	return int64(bytes), nil
}

func rateLimiter(bytesPerSecond int64) *rate.Limiter {
	if bytesPerSecond <= 0 {
		return rate.NewLimiter(rate.Inf, 0)
	}
	// The burst must cover one request; the client asks for up to 16 KiB chunks.
	return rate.NewLimiter(rate.Limit(bytesPerSecond), int(max(bytesPerSecond, 256<<10)))
}

// newTorrentClient starts a client whose files live under dataDir. upload enables
// sharing pieces with other peers; otherwise the client only downloads.
func newTorrentClient(dataDir string, options torrentOptions, upload bool) (*torrent.Client, error) {
	config := torrent.NewDefaultClientConfig()
	config.DataDir = dataDir
	// Every torrent is added with its own storage; the default one must not open a
	// second progress database in the same directory.
	config.DefaultStorage = storage.NewFileOpts(storage.NewFileClientOpts{ClientBaseDir: dataDir, PieceCompletion: storage.NewMapPieceCompletion()})
	config.ListenPort = options.ListenPort
	config.NoDHT = options.NoDHT
	config.NoUpload = !upload
	config.Seed = upload
	config.UploadRateLimiter = rateLimiter(options.UploadLimit)
	config.DownloadRateLimiter = rateLimiter(options.DownloadLimit)
	config.Logger = quietTorrentLogger()
	// Never ask the router to open ports (UPnP/NAT-PMP).
	config.NoDefaultPortForwarding = true
	client, err := torrent.NewClient(config)
	if err != nil {
		if options.ListenPort != 0 {
			return nil, fmt.Errorf("start torrent client on --listen-port %d: %w", options.ListenPort, err)
		}
		return nil, fmt.Errorf("start torrent client: %w", err)
	}
	return client, nil
}

// quietTorrentLogger discards the torrent library's logging; lmx reports progress itself.
func quietTorrentLogger() alog.Logger {
	logger := alog.Default
	logger.SetHandlers(alog.DiscardHandler)
	return logger
}

// addMagnetPeers adds peers named in the magnet itself (x.pe, BEP 9); trackers and DHT
// find the rest.
func addMagnetPeers(t *torrent.Torrent, magnet string) {
	parsed, err := metainfo.ParseMagnetUri(magnet)
	if err != nil {
		return
	}
	var peers []torrent.PeerInfo
	for _, hostPort := range parsed.Params["x.pe"] {
		if addr, err := net.ResolveTCPAddr("tcp", hostPort); err == nil {
			peers = append(peers, torrent.PeerInfo{Addr: addr, Source: torrent.PeerSourceDirect})
		}
	}
	t.AddPeers(peers)
}

// Torrent metadata is cached under the model store once any peer sent it, so seeding a
// model later does not depend on another peer being online.
func torrentMetaPath(root, infohash string) string {
	return filepath.Join(root, "torrent-meta", infohash+".torrent")
}

func saveTorrentMeta(root string, t *torrent.Torrent) {
	path := torrentMetaPath(root, t.InfoHash().HexString())
	if _, err := os.Stat(path); err == nil {
		return
	}
	if err := os.MkdirAll(filepath.Dir(path), 0o755); err != nil {
		return
	}
	mi := t.Metainfo()
	tmp, err := os.CreateTemp(filepath.Dir(path), ".meta-*")
	if err != nil {
		return
	}
	if err := mi.Write(tmp); err != nil || tmp.Close() != nil {
		tmp.Close()
		os.Remove(tmp.Name())
		return
	}
	if err := os.Rename(tmp.Name(), path); err != nil {
		os.Remove(tmp.Name())
	}
}

// torrentSpecFor builds a spec from the magnet, adding cached metadata when present.
func torrentSpecFor(root, infohash, magnet string) (*torrent.TorrentSpec, error) {
	spec, err := torrent.TorrentSpecFromMagnetUri(magnet)
	if err != nil {
		return nil, fmt.Errorf("Pirate Face magnet: %w", err)
	}
	if mi, err := metainfo.LoadFromFile(torrentMetaPath(root, infohash)); err == nil && mi.HashInfoBytes() == spec.InfoHash {
		spec.InfoBytes = mi.InfoBytes
	}
	return spec, nil
}

// pieceCompletionFor stores piece progress for downloads (resumable across runs).
func pieceCompletionFor(dir string) (storage.PieceCompletion, error) {
	completion, err := storage.NewBoltPieceCompletion(dir)
	if err != nil {
		return nil, fmt.Errorf("torrent progress database: %w", err)
	}
	return completion, nil
}
