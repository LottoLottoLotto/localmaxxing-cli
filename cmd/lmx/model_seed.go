package main

import (
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"os"
	"os/exec"
	"path/filepath"
	"sort"
	"strings"
	"text/tabwriter"
	"time"

	"github.com/anacrolix/torrent"
	"github.com/anacrolix/torrent/metainfo"
	"github.com/anacrolix/torrent/storage"
)

// lmx model seed shares downloaded snapshots with the Pirate Face swarm, straight from
// the model store (no second copy). It never runs unless asked: in the foreground until
// Ctrl-C or --seed-ratio / --seed-time, or in the background with --detach.

// seedTarget is one snapshot to seed: torrent file paths mapped to files in the store.
type seedTarget struct {
	Repo       string            `json:"repo"`
	Revision   string            `json:"revision"`
	SnapshotID string            `json:"snapshotId"`
	Infohash   string            `json:"infohash"`
	Magnet     string            `json:"-"`
	Files      map[string]string `json:"-"`
	Bytes      int64             `json:"bytes"`
}

// seedStat is the live state of one seeded torrent.
type seedStat struct {
	Repo          string  `json:"repo"`
	SnapshotID    string  `json:"snapshotId"`
	Infohash      string  `json:"infohash"`
	State         string  `json:"state"` // waiting_for_metadata, checking, seeding, done, unavailable
	Reason        string  `json:"reason,omitempty"`
	PiecesReady   int     `json:"piecesReady"`
	Pieces        int     `json:"pieces"`
	UploadedBytes int64   `json:"uploadedBytes"`
	Ratio         float64 `json:"ratio"`
	Peers         int     `json:"peers"`
}

func seedingRoot() (string, error) {
	loaded, err := loadedRoot()
	if err != nil {
		return "", err
	}
	return filepath.Join(filepath.Dir(loaded), "seeding"), nil
}

// seedTargets resolves snapshots (by ID, ID prefix or repository; every ready snapshot
// with --all) to seedable torrents. Snapshots Pirate Face doesn't list, or lists at
// another revision, are reported and skipped.
func seedTargets(args cliArgs, root, target string) ([]seedTarget, error) {
	snapshots, err := modelSnapshotInventory(root)
	if err != nil {
		return nil, err
	}
	var selected []modelSnapshot
	for _, snapshot := range snapshots {
		if snapshot.State != "ready" {
			continue
		}
		if hasFlag(args, "all") || snapshot.ID == target || strings.EqualFold(snapshot.Repo, target) || (len(target) >= 6 && strings.HasPrefix(snapshot.ID, target)) {
			selected = append(selected, snapshot)
		}
	}
	if len(selected) == 0 {
		if hasFlag(args, "all") {
			return nil, cliError{"no_snapshots", "There are no downloaded snapshots to seed.", []string{"Download one with lmx model snapshot org/model --source pirateface --yes."}, nil}
		}
		return nil, cliError{"snapshot_not_found", "No downloaded snapshot matches " + target + ".", []string{"Seeding works on snapshots; list them with lmx model snapshots list.", "Download one with lmx model snapshot org/model --source pirateface --yes."}, nil}
	}
	client := modelPullHTTPClient()
	defer client.CloseIdleConnections()
	listings := map[string]*pirateFaceListing{}
	var targets []seedTarget
	for _, snapshot := range selected {
		listing, cached := listings[snapshot.Repo]
		if !cached {
			if listing, err = lookupPirateFace(client, pirateFaceURL(args), snapshot.Repo); err != nil {
				return nil, err
			}
			listings[snapshot.Repo] = listing
		}
		if listing == nil {
			printStatus(args, "seed_skipped", map[string]any{"repo": snapshot.Repo, "snapshot": snapshot.ID, "reason": "Pirate Face does not list this model"})
			continue
		}
		if listing.Revision != "" && listing.Revision != snapshot.Revision {
			printStatus(args, "seed_skipped", map[string]any{"repo": snapshot.Repo, "snapshot": snapshot.ID, "reason": fmt.Sprintf("Pirate Face's torrent is revision %s; this snapshot is %s", listing.Revision[:7], snapshot.Revision[:7])})
			continue
		}
		seed := seedTarget{Repo: snapshot.Repo, Revision: snapshot.Revision, SnapshotID: snapshot.ID, Infohash: listing.Infohash, Magnet: listing.Magnet, Files: map[string]string{}}
		for _, file := range snapshot.Files {
			if file.Artifact != nil {
				seed.Files[file.File] = file.Artifact.Path
				seed.Bytes += file.Size
			}
		}
		targets = append(targets, seed)
	}
	if len(targets) == 0 {
		return nil, cliError{"nothing_to_seed", "None of the selected snapshots can be seeded.", []string{"Pirate Face must list the model at the snapshot's revision; see the seed_skipped messages."}, nil}
	}
	return targets, nil
}

// runSeeding seeds targets until ctx ends, --seed-time passes, or every torrent reached
// --seed-ratio. report receives the stats every few seconds.
func runSeeding(ctx context.Context, args cliArgs, root string, targets []seedTarget, options torrentOptions, report func([]seedStat)) error {
	base, err := seedingRoot()
	if err != nil {
		return err
	}
	if err := os.MkdirAll(base, 0o700); err != nil {
		return err
	}
	work, err := os.MkdirTemp(base, "session-")
	if err != nil {
		return err
	}
	defer os.RemoveAll(work)
	client, err := newTorrentClient(work, options, true)
	if err != nil {
		return err
	}
	defer func() {
		client.Close()
		<-client.Closed()
	}()

	stats := make([]seedStat, len(targets))
	torrents := make([]*torrent.Torrent, len(targets))
	for i, target := range targets {
		stats[i] = seedStat{Repo: target.Repo, SnapshotID: target.SnapshotID, Infohash: target.Infohash, State: "waiting_for_metadata"}
		// A view of symlinks to the store files: the torrent reads them in place.
		view := filepath.Join(work, target.Infohash)
		for name, path := range target.Files {
			link := filepath.Join(view, filepath.FromSlash(name))
			if err := os.MkdirAll(filepath.Dir(link), 0o700); err != nil {
				return err
			}
			if err := os.Symlink(path, link); err != nil {
				return err
			}
		}
		spec, err := torrentSpecFor(root, target.Infohash, target.Magnet)
		if err != nil {
			return err
		}
		spec.Storage = storage.NewFileOpts(storage.NewFileClientOpts{
			ClientBaseDir:   view,
			TorrentDirMaker: func(baseDir string, _ *metainfo.Info, _ metainfo.Hash) string { return baseDir },
			FilePathMaker:   func(opts storage.FilePathMakerOpts) string { return filepath.Join(opts.File.BestPath()...) },
			PieceCompletion: storage.NewMapPieceCompletion(),
		})
		t, _, err := client.AddTorrentSpec(spec)
		if err != nil {
			return fmt.Errorf("%s: %w", target.Repo, err)
		}
		// Seeding only ever reads the store: nothing is downloaded into it.
		t.DisallowDataDownload()
		addMagnetPeers(t, target.Magnet)
		torrents[i] = t
	}

	started := time.Now()
	tick := time.NewTicker(2 * time.Second)
	defer tick.Stop()
	lastReport := time.Time{}
	for {
		active := 0
		for i, t := range torrents {
			stat := &stats[i]
			switch stat.State {
			case "waiting_for_metadata":
				select {
				case <-t.GotInfo():
					saveTorrentMeta(root, t)
					stat.State, stat.Pieces = "checking", t.NumPieces()
					t.VerifyData()
				default:
				}
			case "checking":
				if pieceChecksPending(t) {
					break
				}
				stat.PiecesReady = t.Stats().PiecesComplete
				if stat.PiecesReady == 0 {
					stat.State, stat.Reason = "unavailable", "the local files do not match the torrent"
					t.Drop()
					break
				}
				stat.State = "seeding"
				printStatus(args, "seed_ready", map[string]any{"repo": stat.Repo, "pieces": fmt.Sprintf("%d/%d", stat.PiecesReady, stat.Pieces)})
			case "seeding":
				s := t.Stats()
				stat.UploadedBytes = s.BytesWrittenData.Int64()
				stat.Peers = s.ActivePeers
				if targets[i].Bytes > 0 {
					stat.Ratio = float64(stat.UploadedBytes) / float64(targets[i].Bytes)
				}
				if options.SeedRatio > 0 && stat.Ratio >= options.SeedRatio {
					stat.State, stat.Reason = "done", fmt.Sprintf("reached --seed-ratio %g", options.SeedRatio)
					t.Drop()
				}
			}
			if stat.State == "waiting_for_metadata" || stat.State == "checking" || stat.State == "seeding" {
				active++
			}
		}
		timeUp := options.SeedTime > 0 && time.Since(started) >= options.SeedTime
		if time.Since(lastReport) >= 10*time.Second || active == 0 || timeUp {
			report(stats)
			lastReport = time.Now()
		}
		if active == 0 {
			return nil
		}
		if timeUp {
			for i := range stats {
				if stats[i].State == "seeding" || stats[i].State == "waiting_for_metadata" || stats[i].State == "checking" {
					stats[i].State, stats[i].Reason = "done", fmt.Sprintf("reached --seed-time %s", options.SeedTime)
				}
			}
			report(stats)
			return nil
		}
		select {
		case <-ctx.Done():
			report(stats)
			return nil
		case <-tick.C:
		}
	}
}

func pieceChecksPending(t *torrent.Torrent) bool {
	for _, run := range t.PieceStateRuns() {
		if run.Checking || run.Hashing || run.QueuedForHash || run.Marking {
			return true
		}
	}
	return false
}

func printSeedStats(args cliArgs, stats []seedStat) {
	for _, stat := range stats {
		printStatus(args, "seeding_progress", map[string]any{
			"repo": stat.Repo, "state": stat.State, "reason": stat.Reason, "peers": stat.Peers,
			"uploadedMiB": stat.UploadedBytes >> 20, "ratio": fmt.Sprintf("%.2f", stat.Ratio),
		})
	}
}

// seedingRecord describes a background seeding process; its stats file is written by
// the process itself.
type seedingRecord struct {
	Name            string       `json:"name"`
	Targets         []seedTarget `json:"targets"`
	PID             int          `json:"pid"`
	ProcessIdentity string       `json:"processIdentity"`
	LogPath         string       `json:"logPath"`
	StartedAt       string       `json:"startedAt"`
	Argv            []string     `json:"argv"`
}

type seedingStatsFile struct {
	UpdatedAt string     `json:"updatedAt"`
	Torrents  []seedStat `json:"torrents"`
}

func seedingRecordName(args cliArgs, target string) string {
	if hasFlag(args, "all") {
		return "all"
	}
	return loadedName(target)
}

func handleModelSeed(target string, args cliArgs) error {
	if target == "" && !hasFlag(args, "all") {
		return cliError{"missing_model", "model seed requires a snapshot ID or repository (or --all).", []string{"List snapshots with lmx model snapshots list."}, nil}
	}
	options, err := parseTorrentOptions(args)
	if err != nil {
		return err
	}
	root, err := modelStoreRoot(args)
	if err != nil {
		return err
	}
	targets, err := seedTargets(args, root, target)
	if err != nil {
		return err
	}
	name := firstNonEmpty(opt(args, "seeding-name"), seedingRecordName(args, target))
	if hasFlag(args, "dry-run") {
		return writeOrPrintJSON("model_seed_plan", args, map[string]any{"name": name, "targets": targets, "uploadLimit": options.UploadLimit, "seedRatio": options.SeedRatio, "seedTime": options.SeedTime.String()})
	}
	if hasFlag(args, "detach") {
		return startDetachedSeeding(args, name, targets)
	}
	ctx, stop := terminalSignalContext()
	defer stop()
	report := func(stats []seedStat) { printSeedStats(args, stats) }
	if opt(args, "seeding-name") != "" {
		// Background child: also publish stats for lmx model seeding.
		report = func(stats []seedStat) {
			printSeedStats(args, stats)
			_ = writeSeedingStats(name, stats)
		}
	} else {
		printStatus(args, "seeding_start", map[string]any{"torrents": len(targets), "stop": "Ctrl-C"})
	}
	return runSeeding(ctx, args, root, targets, options, report)
}

func seedingPaths(name string) (record, stats, logPath string, err error) {
	root, err := seedingRoot()
	if err != nil {
		return "", "", "", err
	}
	return filepath.Join(root, name+".json"), filepath.Join(root, name+".stats.json"), filepath.Join(root, name+".log"), nil
}

func writeSeedingStats(name string, stats []seedStat) error {
	_, path, _, err := seedingPaths(name)
	if err != nil {
		return err
	}
	return writeJSONAtomic(path, seedingStatsFile{UpdatedAt: time.Now().UTC().Format(time.RFC3339), Torrents: stats})
}

func readSeedingRecords() ([]seedingRecord, error) {
	root, err := seedingRoot()
	if err != nil {
		return nil, err
	}
	files, err := filepath.Glob(filepath.Join(root, "*.json"))
	if err != nil {
		return nil, err
	}
	records := []seedingRecord{}
	for _, file := range files {
		if strings.HasSuffix(file, ".stats.json") {
			continue
		}
		var record seedingRecord
		data, err := os.ReadFile(file)
		if err != nil || json.Unmarshal(data, &record) != nil || !loadedNamePattern.MatchString(record.Name) {
			continue
		}
		records = append(records, record)
	}
	sort.Slice(records, func(i, j int) bool { return records[i].Name < records[j].Name })
	return records, nil
}

func (r seedingRecord) alive() bool {
	return r.PID > 0 && processMatchesIdentity(r.PID, r.ProcessIdentity)
}

func startDetachedSeeding(args cliArgs, name string, targets []seedTarget) error {
	recordPath, statsPath, logPath, err := seedingPaths(name)
	if err != nil {
		return err
	}
	if data, err := os.ReadFile(recordPath); err == nil {
		var existing seedingRecord
		if json.Unmarshal(data, &existing) == nil && existing.alive() {
			return cliError{"already_seeding", name + " is already seeding in the background.", []string{"Stop it with lmx model unseed " + name + "."}, nil}
		}
	}
	if err := os.MkdirAll(filepath.Dir(recordPath), 0o700); err != nil {
		return err
	}
	logFile, err := os.OpenFile(logPath, os.O_CREATE|os.O_WRONLY|os.O_TRUNC, 0o600)
	if err != nil {
		return err
	}
	defer logFile.Close()
	executable, err := os.Executable()
	if err != nil {
		return err
	}
	childArgs := terminalArgsWithoutFlags(args, "detach", "json", "quiet", "dry-run")
	childArgs.opts["seeding-name"] = name
	childArgs.flags["json-status"] = true
	argv := terminalArgsToArgv(childArgs)
	cmd := exec.Command(executable, argv...)
	cmd.Stdin, cmd.Stdout, cmd.Stderr, cmd.Env = nil, logFile, logFile, os.Environ()
	configureDetachedCommand(cmd)
	os.Remove(statsPath)
	if err := cmd.Start(); err != nil {
		return fmt.Errorf("start background seeding: %w", err)
	}
	record := seedingRecord{Name: name, Targets: targets, PID: cmd.Process.Pid, LogPath: logPath, StartedAt: time.Now().UTC().Format(time.RFC3339), Argv: append([]string{"lmx"}, argv...)}
	if record.ProcessIdentity, err = captureProcessIdentity(record.PID); err != nil {
		_ = signalDetachedProcess(record.PID, true)
		return fmt.Errorf("could not record the seeding process: %w", err)
	}
	if err := writeJSONAtomic(recordPath, record); err != nil {
		_ = signalDetachedProcess(record.PID, true)
		return err
	}
	_ = cmd.Process.Release()
	if !humanOutput(args) || opt(args, "out") != "" {
		return writeOrPrintJSON("model_seed", args, map[string]any{"status": "seeding", "seeding": record})
	}
	repos := make([]string, len(targets))
	for i, target := range targets {
		repos[i] = target.Repo
	}
	fmt.Printf("Seeding %s in the background as %q.\n  status: lmx model seeding\n  log:    %s\n  stop:   lmx model unseed %s\n", strings.Join(repos, ", "), name, logPath, name)
	return nil
}

type seedingStatus struct {
	seedingRecord
	Status    string     `json:"status"`
	UpdatedAt string     `json:"updatedAt,omitempty"`
	Torrents  []seedStat `json:"torrents"`
}

func handleModelSeeding(args cliArgs) error {
	records, err := readSeedingRecords()
	if err != nil {
		return err
	}
	rows := make([]seedingStatus, 0, len(records))
	for _, record := range records {
		row := seedingStatus{seedingRecord: record, Status: "running"}
		if !record.alive() {
			row.Status = "exited"
		}
		if _, statsPath, _, err := seedingPaths(record.Name); err == nil {
			if data, err := os.ReadFile(statsPath); err == nil {
				var stats seedingStatsFile
				if json.Unmarshal(data, &stats) == nil {
					row.UpdatedAt, row.Torrents = stats.UpdatedAt, stats.Torrents
				}
			}
		}
		rows = append(rows, row)
	}
	if !humanOutput(args) || opt(args, "out") != "" {
		return writeOrPrintJSON("model_seeding", args, map[string]any{"seeding": rows})
	}
	if len(rows) == 0 {
		fmt.Println("Nothing is seeding. Start with lmx model seed <snapshot or org/model> --detach.")
		return nil
	}
	writer := tabwriter.NewWriter(os.Stdout, 0, 4, 2, ' ', 0)
	fmt.Fprintln(writer, "NAME\tMODEL\tSTATE\tPEERS\tUPLOADED\tRATIO\tPID\tUP")
	for _, row := range rows {
		up := "-"
		if started, err := time.Parse(time.RFC3339, row.StartedAt); err == nil && row.Status == "running" {
			up = time.Since(started).Round(time.Second).String()
		}
		if len(row.Torrents) == 0 {
			fmt.Fprintf(writer, "%s\t%s\t%s\t-\t-\t-\t%d\t%s\n", row.Name, seedingRepos(row.Targets), row.Status, row.PID, up)
			continue
		}
		for i, torrent := range row.Torrents {
			state := torrent.State
			if row.Status == "exited" {
				state = "exited"
			}
			name, pid := row.Name, fmt.Sprint(row.PID)
			if i > 0 {
				name, pid, up = "", "", ""
			}
			fmt.Fprintf(writer, "%s\t%s\t%s\t%d\t%s\t%.2f\t%s\t%s\n", name, torrent.Repo, state, torrent.Peers, humanBytes(torrent.UploadedBytes), torrent.Ratio, pid, up)
		}
	}
	writer.Flush()
	return nil
}

func seedingRepos(targets []seedTarget) string {
	repos := make([]string, len(targets))
	for i, target := range targets {
		repos[i] = target.Repo
	}
	return strings.Join(repos, ",")
}

func humanBytes(n int64) string {
	switch {
	case n >= 1<<30:
		return fmt.Sprintf("%.1f GiB", float64(n)/(1<<30))
	case n >= 1<<20:
		return fmt.Sprintf("%.1f MiB", float64(n)/(1<<20))
	case n >= 1<<10:
		return fmt.Sprintf("%.1f KiB", float64(n)/(1<<10))
	}
	return fmt.Sprintf("%d B", n)
}

func handleModelUnseed(target string, args cliArgs) error {
	records, err := readSeedingRecords()
	if err != nil {
		return err
	}
	var selected []seedingRecord
	for _, record := range records {
		if hasFlag(args, "all") || record.Name == target || strings.EqualFold(seedingRepos(record.Targets), target) {
			selected = append(selected, record)
		}
	}
	if len(selected) == 0 {
		if target == "" && !hasFlag(args, "all") {
			return cliError{"missing_model", "model unseed requires a seeding name (or --all).", []string{"Run lmx model seeding."}, nil}
		}
		return cliError{"not_seeding", "Nothing named " + target + " is seeding.", []string{"Run lmx model seeding."}, nil}
	}
	stopped := []string{}
	for _, record := range selected {
		if record.alive() {
			if err := signalDetachedProcess(record.PID, false); err != nil && !errors.Is(err, os.ErrProcessDone) {
				return err
			}
			for i := 0; i < 100 && record.alive(); i++ {
				time.Sleep(100 * time.Millisecond)
			}
			if record.alive() {
				_ = signalDetachedProcess(record.PID, true)
			}
		}
		recordPath, statsPath, _, err := seedingPaths(record.Name)
		if err != nil {
			return err
		}
		os.Remove(recordPath)
		os.Remove(statsPath)
		stopped = append(stopped, record.Name)
	}
	printInfo(args, "model_unseeded", map[string]any{"stopped": strings.Join(stopped, ", ")})
	return nil
}

// seedAfterDownload seeds what a pull or snapshot just stored, for --seed-ratio /
// --seed-time, in the foreground.
func seedAfterDownload(args cliArgs, root string, listing *pirateFaceListing, files map[string]modelArtifact) error {
	options, err := parseTorrentOptions(args)
	if err != nil || !options.seedAfterDownload() {
		return err
	}
	if listing == nil {
		printStatus(args, "seed_skipped", map[string]any{"reason": "Pirate Face does not list this model"})
		return nil
	}
	target := seedTarget{Infohash: listing.Infohash, Magnet: listing.Magnet, Files: map[string]string{}}
	for name, artifact := range files {
		target.Repo, target.Revision = artifact.Repo, artifact.Revision
		if listing.Revision != "" && listing.Revision != artifact.Revision {
			printStatus(args, "seed_skipped", map[string]any{"reason": "Pirate Face's torrent is another revision"})
			return nil
		}
		target.Files[name] = artifact.Path
		target.Bytes += artifact.Size
	}
	ctx, stop := terminalSignalContext()
	defer stop()
	printStatus(args, "seeding_start", map[string]any{"repo": target.Repo, "seedRatio": options.SeedRatio, "seedTime": options.SeedTime.String(), "stop": "Ctrl-C"})
	return runSeeding(ctx, args, root, []seedTarget{target}, options, func(stats []seedStat) { printSeedStats(args, stats) })
}
