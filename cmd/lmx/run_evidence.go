package main

import (
	"bufio"
	"context"
	"encoding/json"
	"io"
	"math"
	"net/http"
	"strconv"
	"strings"
	"time"
)

// Evidence captured with every remote speed test so the server can verify a
// run against its canonical prompt set (see POST /api/speed-tests).
const (
	promptSampleMaxChars   = 2000
	outputSampleMaxChars   = 4000
	outputSampleHeadChars  = 3000
	outputSampleTailChars  = 1000
	engineTimingsMaxBytes  = 8192
	outputSampleEllipsis   = "\n…\n"
	specMetricsScrapeLimit = 4 << 20
)

func truncateRunes(text string, limit int) string {
	if limit <= 0 {
		return ""
	}
	count := 0
	for index := range text {
		if count == limit {
			return text[:index]
		}
		count++
	}
	return text
}

func tailRunes(text string, limit int) string {
	runes := []rune(text)
	if len(runes) <= limit {
		return text
	}
	return string(runes[len(runes)-limit:])
}

// outputSampleText keeps the head and tail of long generations so both the
// opening and the closing of the answer stay inspectable.
func outputSampleText(text string) string {
	if len([]rune(text)) <= outputSampleMaxChars {
		return text
	}
	return truncateRunes(text, outputSampleHeadChars) + outputSampleEllipsis + tailRunes(text, outputSampleTailChars)
}

// mergeEngineTimings combines the endpoint's usage object and llama.cpp's
// timings object into one raw evidence blob. When the JSON would exceed the
// server limit, timings are dropped first, then usage.
func mergeEngineTimings(parts map[string]any) map[string]any {
	merged := map[string]any{}
	for key, value := range parts {
		if value == nil {
			continue
		}
		if obj, ok := value.(map[string]any); ok && len(obj) == 0 {
			continue
		}
		merged[key] = value
	}
	if len(merged) == 0 {
		return nil
	}
	for _, drop := range []string{"timings", "usage"} {
		data, err := json.Marshal(merged)
		if err == nil && len(data) <= engineTimingsMaxBytes {
			return merged
		}
		delete(merged, drop)
		if len(merged) == 0 {
			return nil
		}
	}
	if data, err := json.Marshal(merged); err != nil || len(data) > engineTimingsMaxBytes {
		return nil
	}
	return merged
}

// addRunEvidence writes the verification evidence fields into metrics.
// prompt is the canonical text without the cache-bust nonce; requestPrompt is
// what was actually sent on the last timed request.
func addRunEvidence(metrics map[string]any, prompt, requestPrompt, outputText string, engineTimingsRaw map[string]any) {
	metrics["promptSha256"] = sha256Hex(prompt)
	metrics["promptSample"] = truncateRunes(requestPrompt, promptSampleMaxChars)
	metrics["outputSha256"] = sha256Hex(outputText)
	metrics["outputSample"] = outputSampleText(outputText)
	if engineTimingsRaw != nil {
		metrics["engineTimingsRaw"] = engineTimingsRaw
	}
}

// specMetrics is a snapshot of engine-side speculative decoding counters.
type specMetrics struct {
	draftTokens        float64
	acceptedTokens     float64
	acceptanceRate     float64
	meanAcceptedLength float64
	hasCounters        bool
	hasAcceptanceRate  bool
	hasMeanAccepted    bool
}

// specCounterSums reads llama.cpp's per-request timings (draft_n and
// draft_n_accepted) into a running total.
func (m *specMetrics) addLlamaTimings(timings map[string]any) {
	if timings == nil {
		return
	}
	draft, haveDraft := jsonNumber(timings["draft_n"])
	accepted, haveAccepted := jsonNumber(timings["draft_n_accepted"])
	if !haveDraft && !haveAccepted {
		return
	}
	m.draftTokens += draft
	m.acceptedTokens += accepted
	m.hasCounters = true
}

// scrapeSpecMetrics reads <baseURL>/metrics (Prometheus text) and sums the
// vLLM / SGLang speculative decoding series, ignoring labels. A missing or
// unparsable endpoint returns ok=false and is not an error.
func scrapeSpecMetrics(baseURL, apiKey string) (specMetrics, bool) {
	ctx, cancel := context.WithTimeout(context.Background(), 5*time.Second)
	defer cancel()
	req, err := http.NewRequestWithContext(ctx, "GET", strings.TrimRight(baseURL, "/")+"/metrics", nil)
	if err != nil {
		return specMetrics{}, false
	}
	if apiKey != "" {
		req.Header.Set("Authorization", "Bearer "+apiKey)
	}
	res, err := http.DefaultClient.Do(req)
	if err != nil {
		return specMetrics{}, false
	}
	defer res.Body.Close()
	if res.StatusCode < 200 || res.StatusCode >= 300 {
		return specMetrics{}, false
	}
	return parseSpecMetrics(bufio.NewScanner(io.LimitReader(res.Body, specMetricsScrapeLimit)))
}

func parseSpecMetricsText(text string) (specMetrics, bool) {
	return parseSpecMetrics(bufio.NewScanner(strings.NewReader(text)))
}

func parseSpecMetrics(scanner *bufio.Scanner) (specMetrics, bool) {
	scanner.Buffer(make([]byte, 64*1024), 1<<20)
	sums := map[string]float64{}
	seen := map[string]bool{}
	for scanner.Scan() {
		line := strings.TrimSpace(scanner.Text())
		if line == "" || strings.HasPrefix(line, "#") {
			continue
		}
		name, value, ok := parsePrometheusSample(line)
		if !ok {
			continue
		}
		switch name {
		case "vllm:spec_decode_num_draft_tokens_total", "vllm:spec_decode_num_accepted_tokens_total",
			"vllm:spec_decode_draft_acceptance_rate", "sglang:spec_accept_length", "sglang:spec_accept_rate":
			sums[name] += value
			seen[name] = true
		}
	}
	metrics := specMetrics{}
	if seen["vllm:spec_decode_num_draft_tokens_total"] || seen["vllm:spec_decode_num_accepted_tokens_total"] {
		metrics.hasCounters = true
		metrics.draftTokens = sums["vllm:spec_decode_num_draft_tokens_total"]
		metrics.acceptedTokens = sums["vllm:spec_decode_num_accepted_tokens_total"]
	}
	if seen["vllm:spec_decode_draft_acceptance_rate"] {
		metrics.hasAcceptanceRate = true
		metrics.acceptanceRate = sums["vllm:spec_decode_draft_acceptance_rate"]
	} else if seen["sglang:spec_accept_rate"] {
		metrics.hasAcceptanceRate = true
		metrics.acceptanceRate = sums["sglang:spec_accept_rate"]
	}
	if seen["sglang:spec_accept_length"] {
		metrics.hasMeanAccepted = true
		metrics.meanAcceptedLength = sums["sglang:spec_accept_length"]
	}
	return metrics, metrics.hasCounters || metrics.hasAcceptanceRate || metrics.hasMeanAccepted
}

// parsePrometheusSample splits `name{labels} value [timestamp]` into name and value.
func parsePrometheusSample(line string) (string, float64, bool) {
	nameEnd := strings.IndexAny(line, "{ \t")
	if nameEnd <= 0 {
		return "", 0, false
	}
	name := line[:nameEnd]
	rest := line[nameEnd:]
	if rest[0] == '{' {
		close := strings.LastIndex(rest, "}")
		if close < 0 {
			return "", 0, false
		}
		rest = rest[close+1:]
	}
	fields := strings.Fields(rest)
	if len(fields) == 0 {
		return "", 0, false
	}
	value, err := strconv.ParseFloat(fields[0], 64)
	if err != nil {
		return "", 0, false
	}
	return name, value, true
}

// specMetricsDelta returns the counters accrued between two scrapes; gauges
// (acceptance rate, mean accepted length) come from the later snapshot.
func specMetricsDelta(before, after specMetrics) specMetrics {
	delta := after
	if before.hasCounters && after.hasCounters {
		delta.draftTokens = after.draftTokens - before.draftTokens
		delta.acceptedTokens = after.acceptedTokens - before.acceptedTokens
		if delta.draftTokens < 0 || delta.acceptedTokens < 0 {
			delta.hasCounters = false
		}
	}
	return delta
}

// perRequest scales accumulated counters to one request's: the mean over the
// n measured requests (gauges are already per-request). A speed-test payload
// reports outputTokens for one request, and the server checks the draft
// counters against it.
func (m specMetrics) perRequest(n int) specMetrics {
	if n > 1 && m.hasCounters {
		m.draftTokens = math.Round(m.draftTokens / float64(n))
		m.acceptedTokens = math.Round(m.acceptedTokens / float64(n))
	}
	return m
}

// gaugesOnly drops cumulative counters: a single scrape without a baseline
// holds the server's lifetime totals, not this run's.
func (m specMetrics) gaugesOnly() specMetrics {
	m.hasCounters, m.draftTokens, m.acceptedTokens = false, 0, 0
	return m
}

// applySpecMetrics records acceptance statistics on engineFlags. Only stats
// that are actually known are written.
func applySpecMetrics(flags map[string]any, metrics specMetrics) {
	if metrics.hasCounters && metrics.draftTokens > 0 {
		flags["specDraftTokens"] = int(metrics.draftTokens)
		flags["specAcceptedTokens"] = int(metrics.acceptedTokens)
		flags["specAcceptanceRate"] = roundSpecRatio(metrics.acceptedTokens / metrics.draftTokens)
		flags["specDecoding"] = true
	}
	if metrics.hasAcceptanceRate && metrics.acceptanceRate >= 0 && metrics.acceptanceRate <= 1 {
		if _, present := flags["specAcceptanceRate"]; !present {
			flags["specAcceptanceRate"] = roundSpecRatio(metrics.acceptanceRate)
			flags["specDecoding"] = true
		}
	}
	if metrics.hasMeanAccepted && metrics.meanAcceptedLength >= 1 {
		flags["specMeanAcceptedLength"] = roundSpecRatio(metrics.meanAcceptedLength)
		flags["specDecoding"] = true
	}
}

func roundSpecRatio(value float64) float64 {
	return float64(int64(value*10000+0.5)) / 10000
}

// printVerificationResult surfaces the server's verified-run verdict after a
// submit or dry-run response.
func printVerificationResult(args cliArgs, response any) {
	obj := asObject(response)
	if obj == nil {
		return
	}
	verified, haveVerified := obj["verifiedRun"].(bool)
	var issues []any
	if verification := asObject(obj["verification"]); verification != nil {
		if value, ok := verification["verified"].(bool); ok {
			verified, haveVerified = value, true
		}
		issues = anySlice(verification["issues"])
	}
	if raw := anySlice(obj["verificationIssues"]); len(raw) > 0 && len(issues) == 0 {
		issues = raw
	}
	if !haveVerified && len(issues) == 0 {
		return
	}
	texts := make([]string, 0, len(issues))
	for _, issue := range issues {
		if text := stringValue(issue); text != "" {
			texts = append(texts, text)
		}
	}
	if verified {
		printInfo(args, "verified_run", map[string]any{"verified": true, "summary": "Verified run: yes"})
		return
	}
	printInfo(args, "verified_run", map[string]any{"verified": false, "issues": strings.Join(texts, "; "), "summary": "Not verified: " + strings.Join(texts, "; ")})
}
