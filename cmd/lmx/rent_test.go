package main

import (
	"bufio"
	"bytes"
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"net"
	"net/http"
	"net/http/httptest"
	"os"
	"strings"
	"sync"
	"sync/atomic"
	"testing"
	"time"

	"github.com/LottoLottoLotto/localmaxxing-cli/internal/relay"
	"github.com/LottoLottoLotto/localmaxxing-cli/internal/relayproto"
)

const (
	rentTestListing = "lst_one"
	rentTestModel   = "qwen3-0.6b"
)

var (
	rentTestInternalSecret  = strings.Repeat("i", 64)
	rentTestAuthorizeSecret = strings.Repeat("a", 64)
	rentTestKey             = "bhk_" + strings.Repeat("0123456789", 4)
)

// rentFixture is an in-process site authorize endpoint, relay and target.
type rentFixture struct {
	t             *testing.T
	site          *httptest.Server
	relay         *relay.Server
	relayHTTP     *httptest.Server // public listener
	relayInternal *httptest.Server // internal listener
	target        *httptest.Server
	authCalls     atomic.Int32
	// authorize decides the site's answer; default: Pro owner of every listing.
	authorize func(apiKey string, listings []string) (int, any)
}

func newRentFixture(t *testing.T, target http.Handler) *rentFixture {
	t.Helper()
	f := &rentFixture{t: t}
	f.authorize = func(apiKey string, listings []string) (int, any) {
		if apiKey != rentTestKey {
			return http.StatusUnauthorized, map[string]any{"code": "unauthorized"}
		}
		out := []map[string]any{}
		for _, id := range listings {
			out = append(out, map[string]any{"id": id, "status": "ACTIVE", "maxConcurrency": 4})
		}
		return http.StatusOK, map[string]any{"ownerId": "owner_1", "plan": "PRO", "listings": out}
	}
	f.site = httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if r.URL.Path != "/api/internal/relay/authorize" || r.Header.Get("Authorization") != "Bearer "+rentTestAuthorizeSecret {
			http.Error(w, "forbidden", http.StatusForbidden)
			return
		}
		f.authCalls.Add(1)
		var body struct {
			APIKey     string   `json:"apiKey"`
			ListingIDs []string `json:"listingIds"`
		}
		_ = json.NewDecoder(r.Body).Decode(&body)
		status, payload := f.authorize(body.APIKey, body.ListingIDs)
		w.Header().Set("Content-Type", "application/json")
		w.WriteHeader(status)
		_ = json.NewEncoder(w).Encode(payload)
	}))
	server, err := relay.New(relay.Config{SiteURL: f.site.URL, InternalSecret: rentTestInternalSecret, AuthorizeSecret: rentTestAuthorizeSecret})
	if err != nil {
		t.Fatal(err)
	}
	f.relay = server
	f.relayHTTP = httptest.NewServer(f.relay.PublicHandler())
	f.relayInternal = httptest.NewServer(f.relay.InternalHandler())
	f.target = httptest.NewServer(target)
	t.Cleanup(func() {
		f.relay.Close()
		f.relayHTTP.Close()
		f.relayInternal.Close()
		f.site.Close()
		f.target.Close()
	})
	return f
}

func (f *rentFixture) hostConfig() rentHostConfig {
	return rentHostConfig{
		RelayURL:       "ws" + strings.TrimPrefix(f.relayHTTP.URL, "http") + relayproto.AgentPath,
		APIKey:         rentTestKey,
		Listings:       []string{rentTestListing},
		Target:         f.target.URL,
		Models:         []string{rentTestModel},
		MaxConcurrency: 4,
		SiteURL:        "https://www.localmaxxing.com",
		BackoffBase:    10 * time.Millisecond,
		BackoffMax:     50 * time.Millisecond,
		DrainTimeout:   5 * time.Second,
	}
}

// startHost runs `rent host` until the test ends and returns its result channel.
func (f *rentFixture) startHost(cfg rentHostConfig) (<-chan error, context.CancelFunc) {
	ctx, cancel := context.WithCancel(context.Background())
	done := make(chan error, 1)
	finished := make(chan struct{})
	go func() {
		done <- runRentHost(ctx, cfg)
		close(finished)
	}()
	f.t.Cleanup(func() {
		cancel()
		select {
		case <-finished:
		case <-time.After(10 * time.Second):
			f.t.Error("rent host did not stop after cancel")
		}
	})
	return done, cancel
}

func (f *rentFixture) agents(listing string) (int, int) {
	req, _ := http.NewRequest(http.MethodGet, f.relayInternal.URL+"/internal/agents?listing="+listing, nil)
	req.Header.Set("Authorization", "Bearer "+rentTestInternalSecret)
	res, err := http.DefaultClient.Do(req)
	if err != nil {
		f.t.Fatal(err)
	}
	defer res.Body.Close()
	var body struct {
		Agents   int `json:"agents"`
		InFlight int `json:"inFlight"`
	}
	_ = json.NewDecoder(res.Body).Decode(&body)
	return body.Agents, body.InFlight
}

func (f *rentFixture) waitAgents(listing string, want int) {
	f.t.Helper()
	deadline := time.Now().Add(5 * time.Second)
	for time.Now().Before(deadline) {
		if got, _ := f.agents(listing); got == want {
			return
		}
		time.Sleep(10 * time.Millisecond)
	}
	got, _ := f.agents(listing)
	f.t.Fatalf("agents for %s = %d, want %d", listing, got, want)
}

func (f *rentFixture) forward(ctx context.Context, listing, method, path, body string) *http.Response {
	f.t.Helper()
	req, err := http.NewRequestWithContext(ctx, http.MethodPost, f.relayInternal.URL+"/internal/forward", strings.NewReader(body))
	if err != nil {
		f.t.Fatal(err)
	}
	req.Header.Set("Authorization", "Bearer "+rentTestInternalSecret)
	req.Header.Set(relayproto.HeaderListing, listing)
	req.Header.Set(relayproto.HeaderSession, "sess_1")
	req.Header.Set(relayproto.HeaderMethod, method)
	req.Header.Set(relayproto.HeaderPath, path)
	req.Header.Set("Content-Type", "application/json")
	res, err := http.DefaultClient.Do(req)
	if err != nil {
		f.t.Fatal(err)
	}
	return res
}

func readErrorCode(t *testing.T, res *http.Response) string {
	t.Helper()
	defer res.Body.Close()
	var body relayproto.ErrorBody
	if err := json.NewDecoder(res.Body).Decode(&body); err != nil {
		t.Fatalf("error body: %v", err)
	}
	return body.Error.Code
}

func chatBody(model string, stream bool) string {
	return fmt.Sprintf(`{"model":%q,"stream":%t,"messages":[{"role":"user","content":"hi"}]}`, model, stream)
}

func TestRentRelayRoundTripStreamsSSE(t *testing.T) {
	release := make(chan struct{})
	var seen atomic.Value
	f := newRentFixture(t, http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if r.URL.Path != "/v1/chat/completions" || r.Method != http.MethodPost {
			http.Error(w, "unexpected "+r.Method+" "+r.URL.Path, http.StatusTeapot)
			return
		}
		seen.Store(r.Header.Clone())
		w.Header().Set("Content-Type", "text/event-stream")
		fmt.Fprint(w, "data: one\n\n")
		w.(http.Flusher).Flush()
		select {
		case <-release:
		case <-time.After(5 * time.Second):
		}
		fmt.Fprint(w, "data: two\n\ndata: [DONE]\n\n")
	}))
	f.startHost(f.hostConfig())
	f.waitAgents(rentTestListing, 1)

	res := f.forward(context.Background(), rentTestListing, "POST", "/v1/chat/completions", chatBody(rentTestModel, true))
	defer res.Body.Close()
	if res.StatusCode != http.StatusOK || res.Header.Get("Content-Type") != "text/event-stream; charset=utf-8" {
		t.Fatalf("status %d content-type %q", res.StatusCode, res.Header.Get("Content-Type"))
	}
	reader := bufio.NewReader(res.Body)
	first := make(chan string, 1)
	go func() {
		line, _ := reader.ReadString('\n')
		first <- line
	}()
	select {
	case line := <-first:
		if line != "data: one\n" {
			t.Fatalf("first line %q", line)
		}
	case <-time.After(3 * time.Second):
		t.Fatal("first SSE event was not flushed through the relay before the stream finished")
	}
	close(release)
	rest, _ := io.ReadAll(reader)
	if !strings.Contains(string(rest), "data: two") || !strings.Contains(string(rest), "[DONE]") {
		t.Fatalf("rest of stream %q", rest)
	}
	headers := seen.Load().(http.Header)
	for _, name := range []string{"Authorization", relayproto.HeaderListing, relayproto.HeaderSession} {
		if headers.Get(name) != "" {
			t.Fatalf("target received %s header", name)
		}
	}

	models := f.forward(context.Background(), rentTestListing, "GET", "/v1/models", "")
	defer models.Body.Close()
	var listing struct {
		Data []struct {
			ID string `json:"id"`
		} `json:"data"`
	}
	if err := json.NewDecoder(models.Body).Decode(&listing); err != nil || len(listing.Data) != 1 || listing.Data[0].ID != rentTestModel {
		t.Fatalf("models = %+v (%v)", listing, err)
	}
}

func TestRentRelayRejectsRoutesAndModels(t *testing.T) {
	var targetHits atomic.Int32
	f := newRentFixture(t, http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		targetHits.Add(1)
		w.WriteHeader(http.StatusOK)
	}))
	f.startHost(f.hostConfig())
	f.waitAgents(rentTestListing, 1)

	if res := f.forward(context.Background(), rentTestListing, "POST", "/admin/models", `{}`); res.StatusCode != http.StatusNotFound || readErrorCode(t, res) != "not_found" {
		t.Fatalf("admin route status %d", res.StatusCode)
	}
	res := f.forward(context.Background(), rentTestListing, "POST", "/v1/chat/completions", chatBody("other-model", false))
	if res.StatusCode != http.StatusBadRequest || readErrorCode(t, res) != "model_not_available" {
		t.Fatalf("model rejection status %d", res.StatusCode)
	}
	if targetHits.Load() != 0 {
		t.Fatalf("target was called %d times", targetHits.Load())
	}
}

func TestRentHostEnforcesAllowlistHeadersAndConcurrency(t *testing.T) {
	block := make(chan struct{})
	entered := make(chan struct{}, 4)
	var seen, seenBody atomic.Value
	target := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		seen.Store(r.Header.Clone())
		body, _ := io.ReadAll(r.Body)
		seenBody.Store(string(body))
		if r.URL.Path != "/v1/embeddings" {
			entered <- struct{}{}
			<-block
		}
		w.Header().Set("Content-Type", "application/json")
		fmt.Fprint(w, `{"ok":true}`)
	}))
	defer target.Close()
	defer close(block)
	host := &rentHost{
		cfg: rentHostConfig{
			Target: target.URL, TargetKey: "inference-key", DiscoveryKey: "admin-key",
			Models: []string{rentTestModel}, MaxConcurrency: 1, Client: &http.Client{}, Status: func(string, map[string]any) {},
		},
		slots: make(chan struct{}, 1), idle: make(chan struct{}, 1),
	}
	open := func(method, path, body string, header http.Header) (*http.Response, error) {
		client, server := net.Pipe()
		host.inFlight.Add(1)
		go host.serve(server)
		req, _ := http.NewRequest(method, "http://lmx-agent"+path, strings.NewReader(body))
		for name, values := range header {
			req.Header[name] = values
		}
		go func() { _ = req.Write(client) }()
		return http.ReadResponse(bufio.NewReader(client), req)
	}
	send := func(method, path, body string, header http.Header) *http.Response {
		res, err := open(method, path, body, header)
		if err != nil {
			t.Fatalf("%s %s: %v", method, path, err)
		}
		return res
	}

	for _, route := range [][2]string{{"POST", "/v1/admin/models"}, {"GET", "/v1/chat/completions"}, {"DELETE", "/v1/models"}, {"POST", "/v1/chat/completions/../../admin"}} {
		if res := send(route[0], route[1], `{}`, nil); res.StatusCode != http.StatusNotFound {
			t.Fatalf("%s %s status %d, want 404", route[0], route[1], res.StatusCode)
		}
	}
	// Parser differentials: the target must see exactly the model checked here.
	for _, body := range []string{
		fmt.Sprintf(`{"model":"unlisted","Model":%q,"input":"x"}`, rentTestModel),
		fmt.Sprintf(`{"MODEL":"unlisted","model":%q,"input":"x"}`, rentTestModel),
		fmt.Sprintf(`{"model":"unlisted","model":%q,"input":"x"}`, rentTestModel),
		fmt.Sprintf(`{"model":%q,"input":"x"} {}`, rentTestModel),
		`["x"]`,
	} {
		if res := send("POST", "/v1/embeddings", body, nil); res.StatusCode != http.StatusBadRequest || readErrorCode(t, res) != "invalid_request" {
			t.Fatalf("body %s: status %d, want 400 invalid_request", body, res.StatusCode)
		}
	}
	if seen.Load() != nil {
		t.Fatal("a rejected body reached the target")
	}

	res := send("POST", "/v1/embeddings", fmt.Sprintf(`{ "model" : %q, "input":"x", "nested":{"Model":1} }`, rentTestModel), http.Header{
		"Content-Type": {"application/json"}, "Authorization": {"Bearer stolen"}, "Cookie": {"a=b"}, "X-Forwarded-For": {"1.2.3.4"},
	})
	body, _ := io.ReadAll(res.Body)
	if res.StatusCode != http.StatusOK || string(body) != `{"ok":true}` {
		t.Fatalf("embeddings status %d body %q", res.StatusCode, body)
	}
	headers := seen.Load().(http.Header)
	if headers.Get("Authorization") != "Bearer inference-key" || headers.Get("Cookie") != "" || headers.Get("X-Forwarded-For") != "" || headers.Get("Content-Type") != "application/json" {
		t.Fatalf("target headers not stripped or wrong target key: %v", headers)
	}
	if got := seenBody.Load().(string); got != fmt.Sprintf(`{"model":%q,"input":"x","nested":{"Model":1}}`, rentTestModel) {
		t.Fatalf("target body %s, want the canonical re-encoding", got)
	}

	// One slot: a blocked request holds it and the next one is refused with 429.
	go func() { _, _ = open("POST", "/v1/chat/completions", chatBody(rentTestModel, false), nil) }()
	select {
	case <-entered:
	case <-time.After(5 * time.Second):
		t.Fatal("blocked request never reached the target")
	}
	if res := send("POST", "/v1/chat/completions", chatBody(rentTestModel, false), nil); res.StatusCode != http.StatusTooManyRequests {
		t.Fatalf("over-limit status %d, want 429", res.StatusCode)
	}
}

func TestRentHostProRequiredStopsWithoutRetry(t *testing.T) {
	f := newRentFixture(t, http.NotFoundHandler())
	f.authorize = func(string, []string) (int, any) {
		return http.StatusForbidden, map[string]any{"code": "pro_required"}
	}
	done, _ := f.startHost(f.hostConfig())
	select {
	case err := <-done:
		var cliErr cliError
		if !errors.As(err, &cliErr) || cliErr.Code != "pro_required" {
			t.Fatalf("err = %v, want pro_required cliError", err)
		}
		if len(cliErr.Hints) == 0 || !strings.Contains(cliErr.Hints[0], "https://www.localmaxxing.com/pro") {
			t.Fatalf("hints = %v", cliErr.Hints)
		}
	case <-time.After(5 * time.Second):
		t.Fatal("rent host kept retrying after pro_required")
	}
	time.Sleep(100 * time.Millisecond)
	if calls := f.authCalls.Load(); calls != 1 {
		t.Fatalf("authorize calls = %d, want 1", calls)
	}
}

func TestRentHostRetriesTransientSiteFailure(t *testing.T) {
	f := newRentFixture(t, http.NotFoundHandler())
	ok := f.authorize
	var calls atomic.Int32
	f.authorize = func(apiKey string, listings []string) (int, any) {
		if calls.Add(1) == 1 {
			return http.StatusInternalServerError, map[string]any{"error": "db down"}
		}
		return ok(apiKey, listings)
	}
	f.startHost(f.hostConfig())
	f.waitAgents(rentTestListing, 1)
	if calls.Load() < 2 {
		t.Fatalf("authorize calls = %d, want a retry", calls.Load())
	}
}

func TestRentRelayRevokeClosesAgent(t *testing.T) {
	f := newRentFixture(t, http.NotFoundHandler())
	done, _ := f.startHost(f.hostConfig())
	f.waitAgents(rentTestListing, 1)

	req, _ := http.NewRequest(http.MethodPost, f.relayInternal.URL+"/internal/agents/revoke", strings.NewReader(`{"ownerId":"owner_1"}`))
	req.Header.Set("Authorization", "Bearer "+rentTestInternalSecret)
	res, err := http.DefaultClient.Do(req)
	if err != nil {
		t.Fatal(err)
	}
	var body struct {
		Revoked int `json:"revoked"`
	}
	_ = json.NewDecoder(res.Body).Decode(&body)
	res.Body.Close()
	if body.Revoked != 1 {
		t.Fatalf("revoked = %d, want 1", body.Revoked)
	}
	select {
	case err := <-done:
		var cliErr cliError
		if !errors.As(err, &cliErr) || cliErr.Code != "pro_required" {
			t.Fatalf("err = %v, want pro_required", err)
		}
	case <-time.After(5 * time.Second):
		t.Fatal("revoked agent did not stop")
	}
	f.waitAgents(rentTestListing, 0)
}

func TestRentRelayNoAgentAndBadSecret(t *testing.T) {
	f := newRentFixture(t, http.NotFoundHandler())
	res := f.forward(context.Background(), "lst_nobody", "POST", "/v1/chat/completions", chatBody(rentTestModel, false))
	if res.StatusCode != http.StatusServiceUnavailable || readErrorCode(t, res) != "no_agent" {
		t.Fatalf("status %d, want 503 no_agent", res.StatusCode)
	}
	req, _ := http.NewRequest(http.MethodGet, f.relayInternal.URL+"/internal/agents?listing=x", nil)
	req.Header.Set("Authorization", "Bearer wrong")
	bad, err := http.DefaultClient.Do(req)
	if err != nil {
		t.Fatal(err)
	}
	if bad.StatusCode != http.StatusUnauthorized {
		t.Fatalf("bad secret status %d, want 401", bad.StatusCode)
	}
	bad.Body.Close()
}

func TestRentRelayCancelPropagatesToTarget(t *testing.T) {
	started := make(chan struct{})
	cancelled := make(chan struct{})
	f := newRentFixture(t, http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		_, _ = io.ReadAll(r.Body) // a server notices a client disconnect only after the body is read
		close(started)
		select {
		case <-r.Context().Done():
			close(cancelled)
		case <-time.After(10 * time.Second):
		}
	}))
	f.startHost(f.hostConfig())
	f.waitAgents(rentTestListing, 1)

	ctx, cancel := context.WithCancel(context.Background())
	go func() {
		req, _ := http.NewRequestWithContext(ctx, http.MethodPost, f.relayInternal.URL+"/internal/forward", strings.NewReader(chatBody(rentTestModel, true)))
		req.Header.Set("Authorization", "Bearer "+rentTestInternalSecret)
		req.Header.Set(relayproto.HeaderListing, rentTestListing)
		req.Header.Set(relayproto.HeaderMethod, "POST")
		req.Header.Set(relayproto.HeaderPath, "/v1/chat/completions")
		if res, err := http.DefaultClient.Do(req); err == nil {
			res.Body.Close()
		}
	}()
	select {
	case <-started:
	case <-time.After(5 * time.Second):
		t.Fatal("target never received the request")
	}
	cancel()
	select {
	case <-cancelled:
	case <-time.After(5 * time.Second):
		t.Fatal("target did not see the renter's cancellation")
	}
	deadline := time.Now().Add(5 * time.Second)
	for time.Now().Before(deadline) {
		if _, inFlight := f.agents(rentTestListing); inFlight == 0 {
			return
		}
		time.Sleep(10 * time.Millisecond)
	}
	t.Fatal("relay still counts the cancelled request in flight")
}

func TestRentConnectProxiesWithCredentials(t *testing.T) {
	var mu sync.Mutex
	var got *http.Request
	var gotBody []byte
	site := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		mu.Lock()
		got = r.Clone(context.Background())
		gotBody, _ = io.ReadAll(r.Body)
		mu.Unlock()
		w.Header().Set("Content-Type", "text/event-stream")
		fmt.Fprint(w, "data: hi\n\n")
	}))
	defer site.Close()
	listener, err := net.Listen("tcp", "127.0.0.1:0")
	if err != nil {
		t.Fatal(err)
	}
	port := listener.Addr().(*net.TCPAddr).Port
	proxy := &httptest.Server{Listener: listener, Config: &http.Server{
		Handler: newRentConnectHandler(site.URL+"/api/rentals/lst_one", rentTestKey, "tok_1", rentConnectHosts("127.0.0.1", port), nil),
	}}
	proxy.Start()
	defer proxy.Close()

	res, err := http.Post(proxy.URL+"/v1/chat/completions", "application/json", bytes.NewBufferString(chatBody(rentTestModel, true)))
	if err != nil {
		t.Fatal(err)
	}
	body, _ := io.ReadAll(res.Body)
	res.Body.Close()
	mu.Lock()
	defer mu.Unlock()
	if string(body) != "data: hi\n\n" || res.Header.Get("Content-Type") != "text/event-stream" {
		t.Fatalf("body %q content-type %q", body, res.Header.Get("Content-Type"))
	}
	if got.URL.Path != "/api/rentals/lst_one/v1/chat/completions" || got.Header.Get("Authorization") != "Bearer "+rentTestKey || got.Header.Get("X-Rental-Token") != "tok_1" {
		t.Fatalf("upstream request %s auth=%q token=%q", got.URL.Path, got.Header.Get("Authorization"), got.Header.Get("X-Rental-Token"))
	}
	if !strings.Contains(string(gotBody), rentTestModel) {
		t.Fatalf("upstream body %q", gotBody)
	}
	mu.Unlock()
	// Wrong routes and methods, encoded traversal, DNS-rebound Host, browsers.
	cases := []struct {
		method, path string
		header       http.Header
		host         string
		want         int
	}{
		{"GET", "/admin", nil, "", http.StatusNotFound},
		{"POST", "/v1/..%2f..%2fkeys", nil, "", http.StatusNotFound},
		{"POST", "/v1/chat/completions/../../../keys", nil, "", http.StatusNotFound},
		{"POST", "/v1/models", nil, "", http.StatusNotFound},
		{"DELETE", "/v1/chat/completions", nil, "", http.StatusNotFound},
		{"POST", "/v1/chat/completions", nil, fmt.Sprintf("attacker.example:%d", port), http.StatusForbidden},
		{"POST", "/v1/chat/completions", http.Header{"Origin": {"https://attacker.example"}}, "", http.StatusForbidden},
		{"POST", "/v1/chat/completions", http.Header{"Origin": {"null"}}, fmt.Sprintf("localhost:%d", port), http.StatusForbidden},
	}
	before := got
	for _, c := range cases {
		conn, err := net.Dial("tcp", proxy.Listener.Addr().String())
		if err != nil {
			t.Fatal(err)
		}
		host := firstNonEmpty(c.host, proxy.Listener.Addr().String())
		raw := fmt.Sprintf("%s %s HTTP/1.1\r\nHost: %s\r\nContent-Length: 2\r\nConnection: close\r\n", c.method, c.path, host)
		for name, values := range c.header {
			raw += name + ": " + values[0] + "\r\n"
		}
		_, _ = io.WriteString(conn, raw+"\r\n{}")
		res, err := http.ReadResponse(bufio.NewReader(conn), nil)
		if err != nil {
			t.Fatal(err)
		}
		res.Body.Close()
		conn.Close()
		if res.StatusCode != c.want {
			t.Fatalf("%s %s host=%s %v: status %d, want %d", c.method, c.path, host, c.header, res.StatusCode, c.want)
		}
	}
	mu.Lock()
	if got != before {
		t.Fatal("a rejected request reached the site")
	}
}

func TestRentRelayURLDerivation(t *testing.T) {
	cases := map[string]string{
		"https://www.localmaxxing.com": "wss://www.localmaxxing.com/relay/v1/agent",
		"http://localhost:3000":        "ws://localhost:3000/relay/v1/agent",
		"http://127.0.0.1:3000":        "ws://127.0.0.1:3000/relay/v1/agent",
	}
	t.Setenv("LMX_RELAY_URL", "")
	for api, want := range cases {
		args := parseArgs([]string{"rent", "host", "x", "--api-url", api})
		got, err := rentRelayURL(args)
		if err != nil || got != want {
			t.Fatalf("%s → %q (%v), want %q", api, got, err, want)
		}
	}
	// ws:// would send the API key in cleartext: only loopback is allowed.
	for _, args := range [][]string{
		{"--api-url", "http://www.localmaxxing.com"},
		{"--relay-url", "ws://relay.example.com/relay/v1/agent"},
		{"--relay-url", "ws://192.168.1.5:8080/relay/v1/agent"},
	} {
		var cliErr cliError
		if _, err := rentRelayURL(parseArgs(append([]string{"rent", "host", "x"}, args...))); !errors.As(err, &cliErr) || cliErr.Code != "insecure_relay_url" {
			t.Fatalf("%v: err = %v, want insecure_relay_url", args, err)
		}
	}
	if got, err := rentRelayURL(parseArgs([]string{"rent", "host", "x", "--relay-url", "ws://[::1]:8080/relay/v1/agent"})); err != nil || got == "" {
		t.Fatalf("loopback ws refused: %v", err)
	}
	args := parseArgs([]string{"rent", "host", "x", "--model", "a", "--model", "b,c"})
	if models := optValues(args, "model"); strings.Join(models, ",") != "a,b,c" {
		t.Fatalf("models = %v", models)
	}
}

func TestRentTargetModelsDefaults(t *testing.T) {
	var instancesStatus atomic.Int32
	var instances atomic.Value
	instancesStatus.Store(http.StatusOK)
	instances.Store(`[{"instance":{"name":"running-a"},"status":"ready"},{"instance":{"name":"warming-b"},"status":"starting"},{"instance":{"name":"old-c"},"status":"exited"},{"instance":{"name":"running-a"},"status":"ready"}]`)
	var gotKey atomic.Value
	substrate := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		switch r.URL.Path {
		case "/substrate/v1/instances":
			gotKey.Store(r.Header.Get("Authorization"))
			w.WriteHeader(int(instancesStatus.Load()))
			fmt.Fprint(w, instances.Load().(string))
		case "/v1/models":
			fmt.Fprint(w, `{"data":[{"id":"running-a"},{"id":"warming-b"},{"id":"old-c"},{"id":"registered-tt-model"}]}`)
		default:
			http.NotFound(w, r)
		}
	}))
	defer substrate.Close()
	plain := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if r.URL.Path != "/v1/models" {
			http.NotFound(w, r)
			return
		}
		fmt.Fprint(w, `{"data":[{"id":"x"},{"id":"y"}]}`)
	}))
	defer plain.Close()

	cfg := rentHostConfig{Target: substrate.URL, DiscoveryKey: "admin-key", TargetKey: "inference-key", Client: &http.Client{}}
	models, err := rentTargetModels(cfg)
	if err != nil || strings.Join(models, ",") != "running-a,warming-b" {
		t.Fatalf("substrate models = %v (%v), want only running instances", models, err)
	}
	if gotKey.Load() != "Bearer admin-key" {
		t.Fatalf("instances auth = %v", gotKey.Load())
	}

	instances.Store(`[{"instance":{"name":"old-c"},"status":"exited"}]`)
	var cliErr cliError
	if _, err := rentTargetModels(cfg); !errors.As(err, &cliErr) || cliErr.Code != "no_running_models" || !strings.Contains(strings.Join(cliErr.Hints, " "), "lmx model load") {
		t.Fatalf("nothing running: err = %v", err)
	}

	instancesStatus.Store(http.StatusUnauthorized)
	if _, err := rentTargetModels(cfg); !errors.As(err, &cliErr) || !strings.Contains(strings.Join(cliErr.Hints, " "), "SUBSTRATE_API_KEY") {
		t.Fatalf("keyed gateway: err = %v", err)
	}

	models, err = rentTargetModels(rentHostConfig{Target: plain.URL, Client: &http.Client{}})
	if err != nil || strings.Join(models, ",") != "x,y" {
		t.Fatalf("plain target models = %v (%v)", models, err)
	}
}

func TestRentListShowsInputCachedOutputPrices(t *testing.T) {
	t.Setenv("HOME", t.TempDir())
	site := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if r.URL.Path != "/api/rentals" {
			http.NotFound(w, r)
			return
		}
		w.Header().Set("Content-Type", "application/json")
		_, _ = io.WriteString(w, `{"listings":[
			{"id":"lst_cached","title":"Cached","modelLabel":"m1","pricePer1kInputTokens":0.5,"pricePer1kCachedInputTokens":0.1,"pricePer1kOutputTokens":2,"_count":{"sessions":3}},
			{"id":"lst_inherit","title":"Inherit","modelLabel":"m2","pricePer1kInputTokens":0.75,"pricePer1kCachedInputTokens":null,"pricePer1kOutputTokens":1.5,"_count":{"sessions":0}},
			{"id":"lst_free","title":"Free","modelLabel":"m3","pricePer1kInputTokens":0,"pricePer1kCachedInputTokens":null,"pricePer1kOutputTokens":0,"_count":{"sessions":1}}
		]}`)
	}))
	defer site.Close()

	run := func(argv ...string) string {
		t.Helper()
		original := os.Stdout
		reader, writer, err := os.Pipe()
		if err != nil {
			t.Fatalf("os.Pipe: %v", err)
		}
		os.Stdout = writer
		code := runCLI(append([]string{"rent", "list", "--api-url", site.URL}, argv...))
		_ = writer.Close()
		os.Stdout = original
		data, _ := io.ReadAll(reader)
		_ = reader.Close()
		if code != 0 {
			t.Fatalf("runCLI exit = %d, output = %s", code, data)
		}
		return string(data)
	}

	rowFields := map[string][]string{}
	var header []string
	for _, line := range strings.Split(run(), "\n") {
		fields := strings.Fields(strings.ReplaceAll(line, "|", " "))
		if len(fields) == 0 {
			continue
		}
		if fields[0] == "ID" {
			header = fields
		} else if strings.HasPrefix(fields[0], "lst_") {
			rowFields[fields[0]] = fields
		}
	}
	if strings.Join(header[:7], " ") != "ID TITLE MODEL IN/1K CACHED/1K OUT/1K SESSIONS" {
		t.Fatalf("header = %v", header)
	}
	for id, want := range map[string]string{
		"lst_cached":  "0.5 0.1 2",
		"lst_inherit": "0.75 0.75 1.5",
		"lst_free":    "0 0 0",
	} {
		if got := strings.Join(rowFields[id][3:6], " "); got != want {
			t.Fatalf("%s prices = %q, want %q (row %v)", id, got, want, rowFields[id])
		}
	}

	var payload struct {
		Listings []map[string]any `json:"listings"`
	}
	if err := json.Unmarshal([]byte(run("--json")), &payload); err != nil {
		t.Fatalf("--json output: %v", err)
	}
	if len(payload.Listings) != 3 || payload.Listings[0]["pricePer1kCachedInputTokens"] != 0.1 {
		t.Fatalf("--json listings = %#v", payload.Listings)
	}
	if value, ok := payload.Listings[1]["pricePer1kCachedInputTokens"]; !ok || value != nil {
		t.Fatalf("--json must pass through a null cached price, got %#v", payload.Listings[1])
	}
}
