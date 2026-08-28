package main

import (
	"archive/tar"
	"bytes"
	"compress/gzip"
	"crypto/sha256"
	"encoding/hex"
	"encoding/json"
	"fmt"
	"io"
	"net/http"
	"os"
	"path/filepath"
	"sort"
	"strconv"
	"strings"
	"time"
)

type terminalPublishUpload struct {
	StorageKey  string `json:"storageKey"`
	ContentType string `json:"contentType"`
	Format      string `json:"format"`
	ByteSize    int64  `json:"byteSize"`
	SHA256      string `json:"sha256"`
	ItemCount   int    `json:"itemCount"`
}

type terminalPublishState struct {
	Slug    string                           `json:"slug"`
	Uploads map[string]terminalPublishUpload `json:"uploads"`
}

func publishTerminalDataset(args cliArgs) error {
	source := positional(args, 3)
	if source == "" {
		return cliError{"missing_option", "eval terminal publish requires an imported task bundle directory.", []string{"Run lmx eval terminal import first, then pass its --out directory."}, nil}
	}
	slug, err := requireOpt(args, "slug")
	if err != nil {
		return err
	}
	name, err := requireOpt(args, "name")
	if err != nil {
		return err
	}
	key := apiKey(args)
	if key == "" {
		return missingAPIKey("--api-key or LMX_API_KEY is required for terminal publishing")
	}
	bundles, err := loadTerminalBundles(source)
	if err != nil {
		return err
	}
	if len(bundles) > 500 {
		return cliError{"terminal_dataset_too_many_tasks", "Terminal datasets are limited to 500 tasks.", nil, map[string]any{"tasks": len(bundles)}}
	}
	if !hasFlag(args, "skip-oracle") {
		printStatus(args, "terminal_publish_oracle_start", map[string]any{"tasks": len(bundles)})
		if err := runTerminalEval(args, true); err != nil {
			return err
		}
	}

	shardCount := len(bundles)
	if shardCount > 5 {
		shardCount = 5
	}
	if value := firstNonEmpty(opt(args, "shard-count"), opt(args, "shards")); value != "" {
		parsed, parseErr := strconv.Atoi(value)
		if parseErr != nil || parsed < 2 || parsed > 13 {
			return cliError{"invalid_option", "--shard-count must be between 2 and 13.", nil, map[string]any{"value": value}}
		}
		shardCount = parsed
	}
	if len(bundles) < shardCount {
		return cliError{"dataset_too_small", "The dataset needs at least one task per shard.", []string{"Reduce --shard-count or add more tasks."}, map[string]any{"tasks": len(bundles), "shards": shardCount}}
	}

	statePath := firstNonEmpty(opt(args, "state"), filepath.Join(".localmaxxing", slug+".terminal-publish.json"))
	state := terminalPublishState{Slug: slug, Uploads: map[string]terminalPublishUpload{}}
	if data, readErr := os.ReadFile(statePath); readErr == nil {
		_ = json.Unmarshal(data, &state)
		if state.Uploads == nil {
			state.Uploads = map[string]terminalPublishUpload{}
		}
	}

	rows := make([]string, 0, len(bundles))
	var totalBytes int64
	for index, bundle := range bundles {
		archive, archiveErr := deterministicTerminalTarGz(bundle)
		if archiveErr != nil {
			return archiveErr
		}
		hashBytes := sha256.Sum256(archive)
		hash := hex.EncodeToString(hashBytes[:])
		upload, ok := state.Uploads[hash]
		if !ok {
			printStatus(args, "terminal_bundle_uploading", map[string]any{"taskId": bundle.Task.ID, "index": index + 1, "total": len(bundles), "byteSize": len(archive)})
			upload, err = uploadTerminalBundle(args, bundle.Task.ID+".tar.gz", archive, hash)
			if err != nil {
				return err
			}
			state.Uploads[hash] = upload
			if err := writeTerminalPublishState(statePath, state); err != nil {
				return err
			}
		} else {
			printStatus(args, "terminal_bundle_reused", map[string]any{"taskId": bundle.Task.ID, "storageKey": upload.StorageKey})
		}
		totalBytes += upload.ByteSize
		preview := strings.TrimSpace(bundle.Task.Instruction)
		if len(preview) > 200 {
			preview = preview[:200]
		}
		row, _ := json.Marshal(map[string]any{
			"question_id":         bundle.Task.ID,
			"bundle_key":          upload.StorageKey,
			"sha256":              upload.SHA256,
			"byteSize":            upload.ByteSize,
			"category":            firstNonEmpty(bundle.Task.Category, opt(args, "category"), "agentic"),
			"instruction_preview": preview,
			"task_json":           bundle.Task,
		})
		rows = append(rows, string(row))
	}
	if totalBytes > 20*1024*1024*1024 {
		return cliError{"terminal_dataset_too_large", "Terminal dataset exceeds the 20 GiB limit.", nil, map[string]any{"byteSize": totalBytes}}
	}

	payload := map[string]any{
		"slug": slug, "name": name, "description": opt(args, "description"),
		"category": firstNonEmpty(opt(args, "category"), "agentic"), "sourceUrl": opt(args, "source-url"),
		"taskType": "agentic_terminal", "shardCount": shardCount,
		"jsonl": strings.Join(rows, "\n") + "\n",
	}
	preflight, err := fetchJSON("POST", apiURL(args)+"/api/benchmarks/import/dry-run", key, payload)
	if err != nil {
		return err
	}
	if hasFlag(args, "dry-run") {
		return writeOrPrintJSON("terminal_publish_dry_run", args, map[string]any{"preflight": preflight, "state": statePath, "tasks": len(bundles), "bytes": totalBytes})
	}
	created, err := fetchJSON("POST", apiURL(args)+"/api/benchmarks/import/submit", key, payload)
	if err != nil {
		return err
	}
	return writeOrPrintJSON("terminal_dataset_submitted", args, map[string]any{"dataset": created, "preflight": preflight, "state": statePath})
}

func deterministicTerminalTarGz(bundle terminalBundle) ([]byte, error) {
	var paths []string
	err := filepath.Walk(bundle.Dir, func(path string, info os.FileInfo, walkErr error) error {
		if walkErr != nil {
			return walkErr
		}
		if info.Mode()&os.ModeSymlink != 0 {
			return cliError{"bundle_invalid", "Terminal bundles may not contain symlinks.", nil, map[string]any{"path": path, "taskId": bundle.Task.ID}}
		}
		if path != bundle.Dir {
			paths = append(paths, path)
		}
		return nil
	})
	if err != nil {
		return nil, err
	}
	sort.Strings(paths)
	var compressed bytes.Buffer
	gz, _ := gzip.NewWriterLevel(&compressed, gzip.BestCompression)
	gz.Header.ModTime = time.Unix(0, 0).UTC()
	tw := tar.NewWriter(gz)
	for _, source := range paths {
		info, statErr := os.Stat(source)
		if statErr != nil {
			return nil, statErr
		}
		rel, _ := filepath.Rel(bundle.Dir, source)
		name := filepath.ToSlash(filepath.Join(bundle.Task.ID, rel))
		header, headerErr := tar.FileInfoHeader(info, "")
		if headerErr != nil {
			return nil, headerErr
		}
		header.Name = name
		header.ModTime = time.Unix(0, 0).UTC()
		header.AccessTime = time.Time{}
		header.ChangeTime = time.Time{}
		header.Uid, header.Gid = 0, 0
		header.Uname, header.Gname = "", ""
		if headerErr = tw.WriteHeader(header); headerErr != nil {
			return nil, headerErr
		}
		if info.Mode().IsRegular() {
			file, openErr := os.Open(source)
			if openErr != nil {
				return nil, openErr
			}
			_, copyErr := io.Copy(tw, file)
			file.Close()
			if copyErr != nil {
				return nil, copyErr
			}
		}
	}
	if err := tw.Close(); err != nil {
		return nil, err
	}
	if err := gz.Close(); err != nil {
		return nil, err
	}
	return compressed.Bytes(), nil
}

func uploadTerminalBundle(args cliArgs, filename string, data []byte, hash string) (terminalPublishUpload, error) {
	metadata := map[string]any{"kind": "terminal-task", "filename": filename, "contentType": "application/gzip", "format": "tar.gz", "byteSize": len(data), "sha256": hash, "itemCount": 1}
	value, err := fetchJSON("POST", apiURL(args)+"/api/benchmarks/storage/upload-url", apiKey(args), metadata)
	if err != nil {
		return terminalPublishUpload{}, err
	}
	obj := asObject(value)
	uploadURL := firstString(obj, "uploadUrl", "url")
	if uploadURL == "" {
		return terminalPublishUpload{}, cliError{"storage_upload_url_missing", "Upload response did not include uploadUrl.", nil, value}
	}
	req, _ := http.NewRequest("PUT", uploadURL, bytes.NewReader(data))
	for key, value := range stringMap(obj["headers"]) {
		if strings.HasPrefix(strings.ToLower(key), "x-amz-meta-") && req.URL.Query().Has(strings.ToLower(key)) {
			continue
		}
		req.Header.Set(key, value)
	}
	res, err := http.DefaultClient.Do(req)
	if err != nil {
		return terminalPublishUpload{}, err
	}
	body, _ := io.ReadAll(res.Body)
	res.Body.Close()
	if res.StatusCode < 200 || res.StatusCode >= 300 {
		return terminalPublishUpload{}, cliError{"storage_put_failed", fmt.Sprintf("Terminal bundle upload failed: %s", res.Status), []string{"Retry; signed upload URLs expire."}, string(body)}
	}
	ref := asObject(obj["storageRef"])
	if ref == nil {
		return terminalPublishUpload{}, cliError{"storage_ref_missing", "Upload response did not include storageRef.", nil, value}
	}
	if _, err := fetchJSON("POST", apiURL(args)+"/api/benchmarks/storage/complete", apiKey(args), map[string]any{"storageRef": ref}); err != nil {
		return terminalPublishUpload{}, err
	}
	return terminalPublishUpload{
		StorageKey: firstString(ref, "storageKey"), ContentType: "application/gzip", Format: "tar.gz",
		ByteSize: int64(len(data)), SHA256: hash, ItemCount: 1,
	}, nil
}

func writeTerminalPublishState(path string, state terminalPublishState) error {
	if err := os.MkdirAll(filepath.Dir(path), 0o755); err != nil {
		return err
	}
	data, _ := json.MarshalIndent(state, "", "  ")
	return os.WriteFile(path, append(data, '\n'), 0o600)
}
