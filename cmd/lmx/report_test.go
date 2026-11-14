package main

import (
	"encoding/json"
	"io"
	"net/http"
	"net/http/httptest"
	"os"
	"path/filepath"
	"strings"
	"testing"
)

func TestReportCreateAndEditUseCanonicalGFMContract(t *testing.T) {
	var createBody map[string]any
	var editBody map[string]any
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if r.Header.Get("Authorization") != "Bearer bhk_test" {
			t.Errorf("Authorization = %q", r.Header.Get("Authorization"))
		}
		switch {
		case r.Method == "POST" && r.URL.Path == "/api/models/Qwen/Qwen3.8-27B/reports":
			if err := json.NewDecoder(r.Body).Decode(&createBody); err != nil {
				t.Error(err)
			}
			w.WriteHeader(http.StatusCreated)
			_, _ = io.WriteString(w, `{"id":"report-1","contentFormat":"gfm","formatVersion":1}`)
		case r.Method == "PATCH" && r.URL.Path == "/api/reports/report-1":
			if err := json.NewDecoder(r.Body).Decode(&editBody); err != nil {
				t.Error(err)
			}
			_, _ = io.WriteString(w, `{"id":"report-1","contentFormat":"gfm"}`)
		default:
			http.NotFound(w, r)
		}
	}))
	defer server.Close()

	dir := t.TempDir()
	contentPath := filepath.Join(dir, "report.md")
	if err := os.WriteFile(contentPath, []byte("## Setup\n\nUses **bold** and `code` in a standardized report."), 0o644); err != nil {
		t.Fatal(err)
	}
	out := filepath.Join(dir, "result.json")
	if err := run([]string{"report", "create", "--api-url", server.URL, "--api-key", "bhk_test", "--model", "Qwen/Qwen3.8-27B", "--title", "Agent report", "--summary", "A standardized report created by an agent.", "--content-file", contentPath, "--engine", "FreeToken", "--quantization", "BF16", "--tags", "moe,bf16,moe", "--benchmark-run-ids", "run-a,run-a,run-b", "--draft", "--out", out}); err != nil {
		t.Fatalf("report create: %v", err)
	}
	if createBody["contentFormat"] != "gfm" || createBody["published"] != false {
		t.Fatalf("create body = %#v", createBody)
	}
	if createBody["content"] != "## Setup\n\nUses **bold** and `code` in a standardized report." {
		t.Fatalf("content = %#v", createBody["content"])
	}
	runs, ok := createBody["benchmarkRunIds"].([]any)
	if !ok || len(runs) != 2 || runs[0] != "run-a" || runs[1] != "run-b" {
		t.Fatalf("benchmarkRunIds = %#v", createBody["benchmarkRunIds"])
	}
	if createBody["engineName"] != "FreeToken" || createBody["quantization"] != "BF16" {
		t.Fatalf("report metadata = %#v", createBody)
	}
	tags, ok := createBody["tags"].([]any)
	if !ok || len(tags) != 2 || tags[0] != "moe" || tags[1] != "bf16" {
		t.Fatalf("tags = %#v", createBody["tags"])
	}

	if err := run([]string{"report", "edit", "report-1", "--api-url", server.URL, "--api-key", "bhk_test", "--content-file", contentPath, "--engine=", "--quantization", "BF16", "--tags=", "--eval-run-ids=", "--out", out}); err != nil {
		t.Fatalf("report edit: %v", err)
	}
	if editBody["content"] != createBody["content"] {
		t.Fatalf("edit content = %#v", editBody["content"])
	}
	if values, ok := editBody["evalRunIds"].([]any); !ok || len(values) != 0 {
		t.Fatalf("evalRunIds = %#v", editBody["evalRunIds"])
	}
	if editBody["engineName"] != "" || editBody["quantization"] != "BF16" {
		t.Fatalf("edit metadata = %#v", editBody)
	}
	if values, ok := editBody["tags"].([]any); !ok || len(values) != 0 {
		t.Fatalf("edit tags = %#v", editBody["tags"])
	}
}

func TestReportResponsesCarryPageURL(t *testing.T) {
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		switch {
		case r.Method == "POST" && r.URL.Path == "/api/models/Liquid AI/LFM3-1.7B/reports":
			// No model object in the response: the page URL falls back to --model.
			w.WriteHeader(http.StatusCreated)
			_, _ = io.WriteString(w, `{"id":"r1"}`)
		case r.Method == "GET" && r.URL.Path == "/api/models/Liquid AI/LFM3-1.7B/reports":
			_, _ = io.WriteString(w, `{"reports":[{"id":"r1","model":{"hfId":"Liquid AI/LFM3-1.7B"}},{"id":"r2"}]}`)
		case r.Method == "POST" && r.URL.Path == "/api/reports/r1/publication":
			_, _ = io.WriteString(w, `{"id":"r1","model":{"hfId":"Liquid AI/LFM3-1.7B"},"publishedAt":"2026-11-14T00:00:00Z"}`)
		case r.Method == "GET" && r.URL.Path == "/api/reports/r9":
			_, _ = io.WriteString(w, `{"id":"r9"}`)
		default:
			http.NotFound(w, r)
		}
	}))
	defer server.Close()

	dir := t.TempDir()
	read := func(name string) map[string]any {
		data, err := os.ReadFile(filepath.Join(dir, name))
		if err != nil {
			t.Fatal(err)
		}
		var value map[string]any
		if err := json.Unmarshal(data, &value); err != nil {
			t.Fatal(err)
		}
		return value
	}
	page := server.URL + "/en/models/Liquid%20AI/LFM3-1.7B/reports/r1"
	base := []string{"--api-url", server.URL, "--api-key", "bhk_test"}

	if err := run(append([]string{"report", "create", "--model", "Liquid AI/LFM3-1.7B", "--title", "Page URL", "--summary", "A report whose response has no model.", "--content", "## Body", "--out", filepath.Join(dir, "create.json")}, base...)); err != nil {
		t.Fatalf("report create: %v", err)
	}
	if got := read("create.json")["url"]; got != page {
		t.Fatalf("create url = %#v, want %q", got, page)
	}

	if err := run(append([]string{"report", "publish", "r1", "--out", filepath.Join(dir, "publish.json")}, base...)); err != nil {
		t.Fatalf("report publish: %v", err)
	}
	if got := read("publish.json")["url"]; got != page {
		t.Fatalf("publish url = %#v, want %q", got, page)
	}

	if err := run(append([]string{"report", "list", "--model", "Liquid AI/LFM3-1.7B", "--out", filepath.Join(dir, "list.json")}, base...)); err != nil {
		t.Fatalf("report list: %v", err)
	}
	reports := read("list.json")["reports"].([]any)
	if got := reports[0].(map[string]any)["url"]; got != page {
		t.Fatalf("list url = %#v, want %q", got, page)
	}
	if got := reports[1].(map[string]any)["url"]; got != server.URL+"/en/models/Liquid%20AI/LFM3-1.7B/reports/r2" {
		t.Fatalf("list fallback url = %#v", got)
	}

	// A response without any model id and no --model gives no URL rather than a wrong one.
	if err := run(append([]string{"report", "show", "r9", "--out", filepath.Join(dir, "show.json")}, base...)); err != nil {
		t.Fatalf("report show: %v", err)
	}
	if _, ok := read("show.json")["url"]; ok {
		t.Fatal("show without a model id must not invent a url")
	}
}

func TestReportFormatAndImageUpload(t *testing.T) {
	var uploadedName, uploadedCaption, uploadedContent, uploadedType string
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		switch {
		case r.Method == "GET" && r.URL.Path == "/api/report-format":
			_, _ = io.WriteString(w, `{"contentFormat":"gfm","version":1}`)
		case r.Method == "POST" && r.URL.Path == "/api/reports/report-1/images":
			if err := r.ParseMultipartForm(1 << 20); err != nil {
				t.Error(err)
			}
			uploadedCaption = r.FormValue("altText")
			file, header, err := r.FormFile("file")
			if err != nil {
				t.Error(err)
				return
			}
			defer file.Close()
			uploadedName = header.Filename
			uploadedType = header.Header.Get("Content-Type")
			data, _ := io.ReadAll(file)
			uploadedContent = string(data)
			w.WriteHeader(http.StatusCreated)
			_, _ = io.WriteString(w, `{"id":"image-1","url":"/api/report-images/image-1"}`)
		default:
			http.NotFound(w, r)
		}
	}))
	defer server.Close()

	dir := t.TempDir()
	out := filepath.Join(dir, "result.json")
	if err := run([]string{"report", "format", "--api-url", server.URL, "--out", out}); err != nil {
		t.Fatalf("report format: %v", err)
	}
	formatData, err := os.ReadFile(out)
	if err != nil || !strings.Contains(string(formatData), `"contentFormat": "gfm"`) {
		t.Fatalf("format output = %q, error = %v", formatData, err)
	}
	imagePath := filepath.Join(dir, "evidence.png")
	if err := os.WriteFile(imagePath, []byte("image-data"), 0o644); err != nil {
		t.Fatal(err)
	}
	if err := run([]string{"report", "image", "upload", "report-1", "--api-url", server.URL, "--api-key", "bhk_test", "--file", imagePath, "--caption", "Measured output", "--sort-order", "2", "--out", out}); err != nil {
		t.Fatalf("report image upload: %v", err)
	}
	if uploadedName != "evidence.png" || uploadedCaption != "Measured output" || uploadedContent != "image-data" || uploadedType != "image/png" {
		t.Fatalf("upload = name %q, caption %q, content %q, type %q", uploadedName, uploadedCaption, uploadedContent, uploadedType)
	}
}

func TestReportCommandsAreAgentDiscoverable(t *testing.T) {
	schema := commandSchema()
	data, err := json.Marshal(schema)
	if err != nil {
		t.Fatal(err)
	}
	text := string(data)
	for _, command := range []string{"report create", "report edit", "report image upload", "report publish"} {
		if !strings.Contains(text, `"name":"`+command+`"`) {
			t.Fatalf("command schema omitted %q", command)
		}
	}
	for _, option := range []string{"content-file", "quantization", "hardware-id", "tags"} {
		if !strings.Contains(text, `"name":"`+option+`"`) {
			t.Fatalf("command schema omitted --%s", option)
		}
	}
}
