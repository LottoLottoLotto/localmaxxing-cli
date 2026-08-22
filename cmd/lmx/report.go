package main

import (
	"bytes"
	"encoding/json"
	"fmt"
	"io"
	"mime"
	"mime/multipart"
	"net/http"
	"net/textproto"
	"net/url"
	"os"
	"path/filepath"
	"strconv"
	"strings"
)

const reportTemplate = `# Report title

## Setup

Describe the model, quantization, engine, hardware, command, and relevant settings.

## Methodology

Explain the prompts, run count, warm-up, measurement window, and controls.

## Results

| Metric | Value |
| --- | ---: |
| Decode | 0 tok/s |

## Observations

- **Finding:** Add the main result.
- **Caveat:** Add known limitations.

## Reproduction

` + "```bash\n# Add the exact command here\n```\n"

func handleReport(action, target string, args cliArgs) error {
	switch action {
	case "format", "schema":
		value, err := fetchJSON("GET", apiURL(args)+"/api/report-format", "", nil)
		if err != nil {
			return err
		}
		return writeOrPrintJSON("report_format", args, value)
	case "init":
		return initializeReportDocument(args)
	case "list", "ls":
		return listModelReports(args)
	case "show", "get":
		return showModelReport(target, args)
	case "create", "submit":
		return createModelReport(args)
	case "edit", "update", "patch":
		return editModelReport(target, args)
	case "publish":
		return setModelReportPublication(target, true, args)
	case "unpublish":
		return setModelReportPublication(target, false, args)
	case "delete", "rm", "remove":
		return deleteModelReport(target, args)
	case "image", "images":
		return handleReportImage(positional(args, 2), positional(args, 3), args)
	default:
		return cliError{"unknown_subcommand", "Unknown report subcommand: " + action, []string{"Run lmx report --help to list supported report commands."}, map[string]any{"subcommand": action}}
	}
}

func initializeReportDocument(args cliArgs) error {
	out := opt(args, "out")
	if out == "" || out == "-" {
		fmt.Print(reportTemplate)
		return nil
	}
	if err := os.WriteFile(out, []byte(reportTemplate), 0o644); err != nil {
		return cliError{"file_write_error", fmt.Sprintf("Could not write %s: %v", out, err), nil, nil}
	}
	printStatus(args, "report_template_written", map[string]any{"path": out, "contentFormat": "gfm", "formatVersion": 1})
	return nil
}

func reportModelParts(args cliArgs) (string, string, error) {
	model := firstNonEmpty(opt(args, "model"), opt(args, "hf-id"))
	parts := strings.Split(model, "/")
	if len(parts) != 2 || strings.TrimSpace(parts[0]) == "" || strings.TrimSpace(parts[1]) == "" {
		return "", "", cliError{"invalid_model", "Report commands require --model <organization/model>.", []string{"Example: --model Qwen/Qwen3.8-27B"}, map[string]any{"model": model}}
	}
	return url.PathEscape(parts[0]), url.PathEscape(parts[1]), nil
}

func modelReportsURL(args cliArgs) (string, error) {
	org, model, err := reportModelParts(args)
	if err != nil {
		return "", err
	}
	return apiURL(args) + "/api/models/" + org + "/" + model + "/reports", nil
}

func readReportContent(args cliArgs, required bool) (string, error) {
	if path := opt(args, "content-file"); path != "" {
		var data []byte
		var err error
		if path == "-" {
			data, err = io.ReadAll(os.Stdin)
		} else {
			data, err = os.ReadFile(path)
		}
		if err != nil {
			return "", cliError{"file_read_error", fmt.Sprintf("Could not read report content from %s: %v", path, err), nil, nil}
		}
		return strings.TrimSpace(string(data)), nil
	}
	if content := opt(args, "content"); content != "" {
		return strings.TrimSpace(content), nil
	}
	if required {
		return "", cliError{"missing_content", "Report content is required.", []string{"Pass --content-file report.md, --content <markdown>, or --content-file - for stdin."}, nil}
	}
	return "", nil
}

func commaSeparatedIDs(value string) []string {
	if strings.TrimSpace(value) == "" {
		return []string{}
	}
	seen := map[string]bool{}
	ids := []string{}
	for _, raw := range strings.Split(value, ",") {
		id := strings.TrimSpace(raw)
		if id != "" && !seen[id] {
			seen[id] = true
			ids = append(ids, id)
		}
	}
	return ids
}

func createModelReport(args cliArgs) error {
	endpoint, err := modelReportsURL(args)
	if err != nil {
		return err
	}
	title, err := requireOpt(args, "title")
	if err != nil {
		return err
	}
	summary, err := requireOpt(args, "summary")
	if err != nil {
		return err
	}
	content, err := readReportContent(args, true)
	if err != nil {
		return err
	}
	body := map[string]any{
		"title": title, "summary": summary, "content": content, "contentFormat": "gfm",
		"benchmarkRunIds": commaSeparatedIDs(opt(args, "benchmark-run-ids")),
		"evalRunIds":      commaSeparatedIDs(opt(args, "eval-run-ids")),
		"published":       !hasFlag(args, "draft"),
	}
	value, err := fetchJSON("POST", endpoint, apiKey(args), body)
	if err != nil {
		return err
	}
	return writeOrPrintJSON("report_created", args, value)
}

func listModelReports(args cliArgs) error {
	endpoint, err := modelReportsURL(args)
	if err != nil {
		return err
	}
	value, err := fetchJSON("GET", endpoint, apiKey(args), nil)
	if err != nil {
		return err
	}
	return writeOrPrintJSON("reports", args, value)
}

func requireReportID(target string) (string, error) {
	id := strings.TrimSpace(target)
	if id == "" {
		return "", cliError{"missing_report_id", "A report ID is required.", []string{"Use lmx report list --model <organization/model> to find report IDs."}, nil}
	}
	return url.PathEscape(id), nil
}

func showModelReport(target string, args cliArgs) error {
	id, err := requireReportID(target)
	if err != nil {
		return err
	}
	value, err := fetchJSON("GET", apiURL(args)+"/api/reports/"+id, apiKey(args), nil)
	if err != nil {
		return err
	}
	return writeOrPrintJSON("report", args, value)
}

func editModelReport(target string, args cliArgs) error {
	id, err := requireReportID(target)
	if err != nil {
		return err
	}
	body := map[string]any{}
	if title := opt(args, "title"); title != "" {
		body["title"] = title
	}
	if summary := opt(args, "summary"); summary != "" {
		body["summary"] = summary
	}
	content, err := readReportContent(args, false)
	if err != nil {
		return err
	}
	if content != "" {
		body["content"] = content
	}
	if _, provided := args.provided["benchmark-run-ids"]; provided {
		body["benchmarkRunIds"] = commaSeparatedIDs(opt(args, "benchmark-run-ids"))
	}
	if _, provided := args.provided["eval-run-ids"]; provided {
		body["evalRunIds"] = commaSeparatedIDs(opt(args, "eval-run-ids"))
	}
	if len(body) == 0 {
		return cliError{"empty_update", "Report edit did not specify any changes.", []string{"Pass --title, --summary, --content-file, --content, --benchmark-run-ids, or --eval-run-ids."}, nil}
	}
	value, err := fetchJSON("PATCH", apiURL(args)+"/api/reports/"+id, apiKey(args), body)
	if err != nil {
		return err
	}
	return writeOrPrintJSON("report_updated", args, value)
}

func setModelReportPublication(target string, published bool, args cliArgs) error {
	id, err := requireReportID(target)
	if err != nil {
		return err
	}
	method := "POST"
	title := "report_published"
	if !published {
		method = "DELETE"
		title = "report_unpublished"
	}
	value, err := fetchJSON(method, apiURL(args)+"/api/reports/"+id+"/publication", apiKey(args), nil)
	if err != nil {
		return err
	}
	return writeOrPrintJSON(title, args, value)
}

func deleteModelReport(target string, args cliArgs) error {
	id, err := requireReportID(target)
	if err != nil {
		return err
	}
	if !hasFlag(args, "yes") {
		return cliError{"confirmation_required", "Deleting a report requires --yes.", []string{"This permanently deletes the report, comments, and images."}, nil}
	}
	value, err := fetchJSON("DELETE", apiURL(args)+"/api/reports/"+id, apiKey(args), nil)
	if err != nil {
		return err
	}
	return writeOrPrintJSON("report_deleted", args, value)
}

func handleReportImage(action, target string, args cliArgs) error {
	switch action {
	case "upload", "add":
		id, err := requireReportID(target)
		if err != nil {
			return err
		}
		path, err := requireOpt(args, "file")
		if err != nil {
			return err
		}
		value, err := uploadReportImage(id, path, opt(args, "caption"), opt(args, "sort-order"), args)
		if err != nil {
			return err
		}
		return writeOrPrintJSON("report_image_uploaded", args, value)
	case "delete", "rm", "remove":
		reportID, err := requireReportID(target)
		if err != nil {
			return err
		}
		imageID, err := requireReportID(positional(args, 4))
		if err != nil {
			return cliError{"missing_image_id", "Image deletion requires a report ID and image ID.", []string{"Run lmx report image delete <reportId> <imageId> --yes."}, nil}
		}
		if !hasFlag(args, "yes") {
			return cliError{"confirmation_required", "Deleting a report image requires --yes.", nil, nil}
		}
		value, err := fetchJSON("DELETE", apiURL(args)+"/api/reports/"+reportID+"/images/"+imageID, apiKey(args), nil)
		if err != nil {
			return err
		}
		return writeOrPrintJSON("report_image_deleted", args, value)
	default:
		return cliError{"unknown_subcommand", "Unknown report image subcommand: " + action, []string{"Use upload or delete."}, nil}
	}
}

func uploadReportImage(reportID, path, caption, sortOrder string, args cliArgs) (any, error) {
	file, err := os.Open(path)
	if err != nil {
		return nil, cliError{"file_read_error", fmt.Sprintf("Could not open %s: %v", path, err), nil, nil}
	}
	defer file.Close()
	var body bytes.Buffer
	writer := multipart.NewWriter(&body)
	filename := strings.ReplaceAll(filepath.Base(path), `"`, "")
	headers := make(textproto.MIMEHeader)
	headers.Set("Content-Disposition", fmt.Sprintf(`form-data; name="file"; filename="%s"`, filename))
	contentType := mime.TypeByExtension(strings.ToLower(filepath.Ext(filename)))
	if contentType == "" {
		contentType = "application/octet-stream"
	}
	headers.Set("Content-Type", contentType)
	part, err := writer.CreatePart(headers)
	if err != nil {
		return nil, err
	}
	if _, err := io.Copy(part, file); err != nil {
		return nil, err
	}
	if caption != "" {
		_ = writer.WriteField("altText", caption)
	}
	if sortOrder != "" {
		if _, err := strconv.Atoi(sortOrder); err != nil {
			return nil, cliError{"invalid_sort_order", "--sort-order must be an integer.", nil, map[string]any{"sortOrder": sortOrder}}
		}
		_ = writer.WriteField("sortOrder", sortOrder)
	}
	if err := writer.Close(); err != nil {
		return nil, err
	}
	req, err := http.NewRequest("POST", apiURL(args)+"/api/reports/"+reportID+"/images", &body)
	if err != nil {
		return nil, err
	}
	req.Header.Set("Content-Type", writer.FormDataContentType())
	if key := apiKey(args); key != "" {
		req.Header.Set("Authorization", "Bearer "+key)
	}
	res, err := apiHTTPClient.Do(req)
	if err != nil {
		return nil, cliError{"network_error", fmt.Sprintf("Could not upload report image: %v", err), nil, nil}
	}
	defer res.Body.Close()
	data, _ := io.ReadAll(res.Body)
	var value any
	if len(data) > 0 && json.Unmarshal(data, &value) != nil {
		value = string(data)
	}
	if res.StatusCode < 200 || res.StatusCode >= 300 {
		return nil, cliError{"api_error", fmt.Sprintf("%d %s: %s", res.StatusCode, res.Status, apiMessage(value, string(data))), nil, value}
	}
	return value, nil
}
