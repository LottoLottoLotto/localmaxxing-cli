package main

import (
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"net/http"
	"os"
	"path/filepath"
	"sync"
	"sync/atomic"
	"time"
)

const (
	modelDownloadDefaultConnections = 8
	modelDownloadMaxConnections     = 32
	modelDownloadAttempts           = 3
)

// Variables so tests can exercise chunking with small files.
var (
	// Files with less than two chunks left use one stream; larger files are
	// split into ranges of at least this size.
	modelDownloadMinChunk int64 = 32 << 20
	// Each stream fsyncs and records its progress this often, bounding what an
	// interruption can lose.
	modelDownloadSyncEvery int64 = 16 << 20
	// Pause before retrying a failed range, multiplied by the attempt number.
	modelDownloadRetryDelay = 2 * time.Second
)

var errRangesUnsupported = errors.New("server ignored a byte-range request")

// modelDownloader runs every HTTP stream of one pull or snapshot. slots caps
// concurrent streams across all files at --download-connections.
type modelDownloader struct {
	// ctx is cancelled by Ctrl-C/SIGTERM: streams record their progress and
	// return, so artifact locks are released and a rerun resumes.
	ctx         context.Context
	client      *http.Client
	connections int
	slots       chan struct{}
	args        cliArgs
	planned     atomic.Int64
	received    atomic.Int64
	// swarm holds files Pirate Face delivered (--source pirateface), or nil.
	swarm *pirateFaceSwarm
}

func newModelDownloader(ctx context.Context, client *http.Client, args cliArgs) (*modelDownloader, error) {
	connections, err := intOption(args, modelDownloadDefaultConnections, 1, "download-connections")
	if err != nil {
		return nil, err
	}
	if connections > modelDownloadMaxConnections {
		return nil, cliError{"invalid_option", fmt.Sprintf("--download-connections must be at most %d", modelDownloadMaxConnections), nil, nil}
	}
	return &modelDownloader{ctx: ctx, client: client, connections: connections, slots: make(chan struct{}, connections), args: args}, nil
}

// reportProgress prints aggregate progress every interval until stopped.
func (d *modelDownloader) reportProgress(interval time.Duration) func() {
	done := make(chan struct{})
	finished := make(chan struct{})
	go func() {
		defer close(finished)
		ticker := time.NewTicker(interval)
		defer ticker.Stop()
		last, lastAt := d.received.Load(), time.Now()
		for {
			select {
			case <-done:
				return
			case now := <-ticker.C:
				received := d.received.Load()
				if received == last {
					continue
				}
				rate := float64(received-last) / now.Sub(lastAt).Seconds() / (1 << 20)
				printStatus(d.args, "model_download_progress", map[string]any{
					"receivedMiB": received >> 20, "plannedMiB": d.planned.Load() >> 20,
					"mibPerSecond": fmt.Sprintf("%.1f", rate), "connections": d.connections,
				})
				last, lastAt = received, now
			}
		}
	}()
	return func() { close(done); <-finished }
}

type modelDownloadChunk struct {
	Start int64 `json:"start"`
	End   int64 `json:"end"`  // exclusive
	Done  int64 `json:"done"` // bytes from Start durably written
}

// modelDownloadState is <part>.chunks: which ranges of a preallocated .part are
// on disk. Its presence marks the .part as range-addressed, not a prefix.
type modelDownloadState struct {
	Size         int64                `json:"size"`
	ExpectedHash string               `json:"expectedHash"`
	Chunks       []modelDownloadChunk `json:"chunks"`
}

func newModelDownloadState(source modelPullSource, prefix int64, connections int) modelDownloadState {
	remaining := source.Size - prefix
	chunk := max(modelDownloadMinChunk, (remaining+int64(connections*4)-1)/int64(connections*4))
	state := modelDownloadState{Size: source.Size, ExpectedHash: source.ExpectedHash}
	if prefix > 0 {
		state.Chunks = append(state.Chunks, modelDownloadChunk{Start: 0, End: prefix, Done: prefix})
	}
	for start := prefix; start < source.Size; start += chunk {
		state.Chunks = append(state.Chunks, modelDownloadChunk{Start: start, End: min(start+chunk, source.Size)})
	}
	return state
}

func readModelDownloadState(path string, source modelPullSource) (*modelDownloadState, error) {
	data, err := os.ReadFile(path)
	if errors.Is(err, os.ErrNotExist) {
		return nil, nil
	}
	if err != nil {
		return nil, err
	}
	var state modelDownloadState
	if err := json.Unmarshal(data, &state); err != nil || state.Size != source.Size || state.ExpectedHash != source.ExpectedHash {
		return nil, errors.New("download progress file does not match this artifact")
	}
	next := int64(0)
	for _, c := range state.Chunks {
		if c.Start != next || c.End <= c.Start || c.Done < 0 || c.Done > c.End-c.Start {
			return nil, errors.New("download progress file is inconsistent")
		}
		next = c.End
	}
	if next != source.Size {
		return nil, errors.New("download progress file does not cover the artifact")
	}
	return &state, nil
}

func writeModelDownloadState(path string, state modelDownloadState) error {
	data, err := json.Marshal(state)
	if err != nil {
		return err
	}
	tmp, err := os.CreateTemp(filepath.Dir(path), filepath.Base(path)+".*")
	if err != nil {
		return err
	}
	defer os.Remove(tmp.Name())
	if _, err := tmp.Write(data); err != nil {
		tmp.Close()
		return err
	}
	if err := tmp.Sync(); err != nil {
		tmp.Close()
		return err
	}
	if err := tmp.Close(); err != nil {
		return err
	}
	return os.Rename(tmp.Name(), path)
}

// fetch downloads source into part. Large files use parallel byte ranges;
// small files, --download-connections 1 and servers without range support use
// one stream. Both resume after interruption.
func (d *modelDownloader) fetch(source modelPullSource, part string) error {
	statePath := part + ".chunks"
	state, err := readModelDownloadState(statePath, source)
	if err != nil {
		// Unusable progress: the .part cannot be trusted as a prefix either.
		os.Remove(part)
		os.Remove(statePath)
		state = nil
	}
	if state == nil {
		prefix := int64(0)
		if info, err := os.Lstat(part); err == nil {
			if !info.Mode().IsRegular() {
				return errors.New("partial download is not a regular file")
			}
			prefix = info.Size()
		} else if !os.IsNotExist(err) {
			return err
		}
		if d.connections == 1 || prefix > source.Size || source.Size-prefix < 2*modelDownloadMinChunk {
			return d.sequential(source, part, prefix)
		}
		fresh := newModelDownloadState(source, prefix, d.connections)
		// Record ranges before growing the file so a sized .part is never
		// mistaken for a complete sequential prefix.
		if err := writeModelDownloadState(statePath, fresh); err != nil {
			return err
		}
		state = &fresh
	}
	err = d.parallel(source, part, statePath, state)
	if errors.Is(err, errRangesUnsupported) {
		os.Remove(statePath)
		os.Remove(part)
		return d.sequential(source, part, 0)
	}
	return err
}

func (d *modelDownloader) sequential(source modelPullSource, part string, prefix int64) error {
	d.planned.Add(max(0, source.Size-prefix))
	d.slots <- struct{}{}
	defer func() { <-d.slots }()
	return downloadModelPullCounted(d.ctx, d.client, source, part, &d.received)
}

func (d *modelDownloader) parallel(source modelPullSource, part, statePath string, state *modelDownloadState) error {
	file, err := os.OpenFile(part, os.O_RDWR|os.O_CREATE, 0o600)
	if err != nil {
		return err
	}
	defer file.Close()
	if err := file.Truncate(source.Size); err != nil {
		return err
	}
	var mu sync.Mutex
	persist := func() error { return writeModelDownloadState(statePath, *state) }
	pending := []int{}
	for i, c := range state.Chunks {
		if left := c.End - c.Start - c.Done; left > 0 {
			pending = append(pending, i)
			d.planned.Add(left)
		}
	}
	ctx, cancel := context.WithCancel(d.ctx)
	defer cancel()
	jobs := make(chan int)
	var firstErr error
	var errOnce sync.Once
	var wg sync.WaitGroup
	for range min(d.connections, len(pending)) {
		wg.Add(1)
		go func() {
			defer wg.Done()
			for index := range jobs {
				if err := d.chunkWithRetries(ctx, source, file, &state.Chunks[index], &mu, persist); err != nil {
					errOnce.Do(func() { firstErr = err; cancel() })
				}
			}
		}()
	}
	for _, index := range pending {
		if ctx.Err() != nil {
			break
		}
		jobs <- index
	}
	close(jobs)
	wg.Wait()
	if firstErr != nil {
		return fmt.Errorf("download interrupted; progress retained for retry: %w", firstErr)
	}
	if err := file.Sync(); err != nil {
		return err
	}
	if err := file.Close(); err != nil {
		return err
	}
	return os.Remove(statePath)
}

func (d *modelDownloader) chunkWithRetries(ctx context.Context, source modelPullSource, file *os.File, chunk *modelDownloadChunk, mu *sync.Mutex, persist func() error) error {
	for attempt := 1; ; attempt++ {
		d.slots <- struct{}{}
		err := d.streamChunk(ctx, source, file, chunk, mu, persist)
		<-d.slots
		if err == nil || errors.Is(err, errRangesUnsupported) || ctx.Err() != nil || attempt == modelDownloadAttempts {
			return err
		}
		select {
		case <-ctx.Done():
			return err
		case <-time.After(time.Duration(attempt) * modelDownloadRetryDelay):
		}
	}
}

// streamChunk fetches the unfinished tail of one range into its place in the
// preallocated file, recording durable progress as it goes.
func (d *modelDownloader) streamChunk(ctx context.Context, source modelPullSource, file *os.File, chunk *modelDownloadChunk, mu *sync.Mutex, persist func() error) error {
	mu.Lock()
	start, end := chunk.Start+chunk.Done, chunk.End
	mu.Unlock()
	if start >= end {
		return nil
	}
	req, err := modelPullRequest(ctx, source.URL)
	if err != nil {
		return err
	}
	req.Header.Set("Range", fmt.Sprintf("bytes=%d-%d", start, end-1))
	response, err := d.client.Do(req)
	if err != nil {
		return err
	}
	defer response.Body.Close()
	if encoding := response.Header.Get("Content-Encoding"); encoding != "" && encoding != "identity" {
		return errors.New("unexpected content encoding for model download")
	}
	switch response.StatusCode {
	case http.StatusPartialContent:
	case http.StatusOK:
		return errRangesUnsupported
	default:
		return fmt.Errorf("HuggingFace download returned HTTP %d", response.StatusCode)
	}
	if !modelPullContentRangeSpan(response.Header.Get("Content-Range"), start, end-1, source.Size) {
		return errors.New("invalid Content-Range for pinned model download")
	}
	if response.ContentLength >= 0 && response.ContentLength != end-start {
		return fmt.Errorf("range Content-Length mismatch: expected %d, got %d", end-start, response.ContentLength)
	}
	record := func(offset int64) error {
		if err := file.Sync(); err != nil {
			return err
		}
		mu.Lock()
		defer mu.Unlock()
		chunk.Done = offset - chunk.Start
		return persist()
	}
	buffer := make([]byte, 1<<20)
	offset, unsynced := start, int64(0)
	for offset < end {
		n, readErr := response.Body.Read(buffer[:min(int64(len(buffer)), end-offset)])
		if n > 0 {
			if _, err := file.WriteAt(buffer[:n], offset); err != nil {
				return err
			}
			offset += int64(n)
			unsynced += int64(n)
			d.received.Add(int64(n))
		}
		if unsynced >= modelDownloadSyncEvery || (offset == end && unsynced > 0) {
			if err := record(offset); err != nil {
				return err
			}
			unsynced = 0
		}
		if readErr == io.EOF {
			break
		}
		if readErr != nil {
			if unsynced > 0 {
				if err := record(offset); err != nil {
					return err
				}
			}
			return readErr
		}
	}
	if offset != end {
		if unsynced > 0 {
			if err := record(offset); err != nil {
				return err
			}
		}
		return fmt.Errorf("range truncated: expected %d bytes, received %d", end-start, offset-start)
	}
	return nil
}
