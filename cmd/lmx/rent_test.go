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
	// renter is the site's renter tag sent with forwarded requests ("" none).
	renter string
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
	if f.renter != "" {
		req.Header.Set(relayproto.HeaderRenter, f.renter)
	}
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

// A free account's agent naming a listing that charges is refused for good: it stops at once with the
// site's message and how to fix it, instead of reconnecting forever.
func TestRentHostFreePlanPaidListingStopsWithoutRetry(t *testing.T) {
	f := newRentFixture(t, http.NotFoundHandler())
	f.authorize = func(string, []string) (int, any) {
		return http.StatusForbidden, map[string]any{"code": "free_plan_paid_listing", "message": "Listing l1 charges for requests"}
	}
	done, _ := f.startHost(f.hostConfig())
	select {
	case err := <-done:
		var cliErr cliError
		if !errors.As(err, &cliErr) || cliErr.Code != "free_plan_paid_listing" {
			t.Fatalf("err = %v, want free_plan_paid_listing cliError", err)
		}
		if !strings.Contains(cliErr.Message, "Listing l1 charges for requests") {
			t.Fatalf("message = %q, want the site's", cliErr.Message)
		}
		if len(cliErr.Hints) == 0 || !strings.Contains(cliErr.Hints[0], "price to 0") {
			t.Fatalf("hints = %v", cliErr.Hints)
		}
	case <-time.After(5 * time.Second):
		t.Fatal("rent host kept retrying after free_plan_paid_listing")
	}
	time.Sleep(100 * time.Millisecond)
	if calls := f.authCalls.Load(); calls != 1 {
		t.Fatalf("authorize calls = %d, want 1", calls)
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
	emx := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		switch r.URL.Path {
		case "/emx/v1/instances":
			gotKey.Store(r.Header.Get("Authorization"))
			w.WriteHeader(int(instancesStatus.Load()))
			fmt.Fprint(w, instances.Load().(string))
		case "/v1/models":
			fmt.Fprint(w, `{"data":[{"id":"running-a"},{"id":"warming-b"},{"id":"old-c"},{"id":"registered-tt-model"}]}`)
		default:
			http.NotFound(w, r)
		}
	}))
	defer emx.Close()
	plain := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if r.URL.Path != "/v1/models" {
			http.NotFound(w, r)
			return
		}
		fmt.Fprint(w, `{"data":[{"id":"x"},{"id":"y"}]}`)
	}))
	defer plain.Close()

	cfg := rentHostConfig{Target: emx.URL, DiscoveryKey: "admin-key", TargetKey: "inference-key", Client: &http.Client{}}
	models, err := rentTargetModels(cfg)
	if err != nil || strings.Join(models, ",") != "running-a,warming-b" {
		t.Fatalf("emx models = %v (%v), want only running instances", models, err)
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
	if _, err := rentTargetModels(cfg); !errors.As(err, &cliErr) || !strings.Contains(strings.Join(cliErr.Hints, " "), "EMX_API_KEY") {
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
			{"id":"lst_cached","title":"Cached","modelLabel":"m1","pricePer1MInputTokens":5,"pricePer1MCachedInputTokens":1,"pricePer1MOutputTokens":20,"_count":{"sessions":3}},
			{"id":"lst_inherit","title":"Inherit","modelLabel":"m2","pricePer1MInputTokens":7.5,"pricePer1MCachedInputTokens":null,"pricePer1MOutputTokens":15,"_count":{"sessions":0}},
			{"id":"lst_free","title":"Free","modelLabel":"m3","pricePer1MInputTokens":0,"pricePer1MCachedInputTokens":null,"pricePer1MOutputTokens":0,"_count":{"sessions":1}}
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
	if strings.Join(header[:7], " ") != "ID TITLE MODEL IN/1M CACHED/1M OUT/1M SESSIONS" {
		t.Fatalf("header = %v", header)
	}
	for id, want := range map[string]string{
		"lst_cached":  "$5 $1 $20",
		"lst_inherit": "$7.5 $7.5 $15",
		"lst_free":    "$0 $0 $0",
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
	if len(payload.Listings) != 3 || payload.Listings[0]["pricePer1MCachedInputTokens"] != 1.0 {
		t.Fatalf("--json listings = %#v", payload.Listings)
	}
	if value, ok := payload.Listings[1]["pricePer1MCachedInputTokens"]; !ok || value != nil {
		t.Fatalf("--json must pass through a null cached price, got %#v", payload.Listings[1])
	}
}

// A host serving two models for two listings: each listing reaches only the models it offers, so a
// listing priced for the small model can never be used for the big one.
func TestRentHostServesOnlyEachListingsModels(t *testing.T) {
	var targetHits atomic.Int32
	f := newRentFixture(t, http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		targetHits.Add(1)
		w.Header().Set("Content-Type", "text/event-stream")
		fmt.Fprint(w, "data: {\"choices\":[{\"delta\":{\"content\":\"ok\"}}]}\n\ndata: [DONE]\n\n")
	}))
	const second = "lst_two"
	authorize := f.authorize
	f.authorize = func(apiKey string, listings []string) (int, any) {
		_, payload := authorize(apiKey, []string{rentTestListing})
		p := payload.(map[string]any)
		p["listings"] = []map[string]any{{"id": rentTestListing, "status": "ACTIVE", "maxConcurrency": 4}, {"id": second, "status": "ACTIVE", "maxConcurrency": 4}}
		return http.StatusOK, p
	}
	cfg := f.hostConfig()
	cfg.Listings = []string{rentTestListing, second}
	cfg.Models = []string{rentTestModel, "qwen3.8-27b"}
	cfg.Offers = &rentListingOffers{}
	cfg.Offers.set(map[string][]string{rentTestListing: {rentTestModel}, second: {"qwen3.8-27b", "not-served-here"}})
	f.startHost(cfg)
	f.waitAgents(rentTestListing, 1)

	if res := f.forward(context.Background(), rentTestListing, "POST", "/v1/chat/completions", chatBody(rentTestModel, true)); res.StatusCode != http.StatusOK {
		t.Fatalf("the listing's own model: %d", res.StatusCode)
	}
	res := f.forward(context.Background(), rentTestListing, "POST", "/v1/chat/completions", chatBody("qwen3.8-27b", true))
	if res.StatusCode != http.StatusBadRequest || readErrorCode(t, res) != "model_not_available" {
		t.Fatalf("another listing's model through this listing: %d", res.StatusCode)
	}
	if res := f.forward(context.Background(), second, "POST", "/v1/chat/completions", chatBody("qwen3.8-27b", true)); res.StatusCode != http.StatusOK {
		t.Fatalf("the second listing's model: %d", res.StatusCode)
	}
	res = f.forward(context.Background(), second, "POST", "/v1/chat/completions", chatBody("not-served-here", true))
	if res.StatusCode != http.StatusBadRequest || readErrorCode(t, res) != "model_not_available" {
		t.Fatalf("offered by the listing but not served here: %d", res.StatusCode)
	}
	if n := targetHits.Load(); n != 2 {
		t.Fatalf("the target saw %d requests, want the 2 allowed ones", n)
	}
	models := f.forward(context.Background(), second, "GET", "/v1/models", "")
	defer models.Body.Close()
	var list struct {
		Data []struct {
			ID string `json:"id"`
		} `json:"data"`
	}
	if err := json.NewDecoder(models.Body).Decode(&list); err != nil || len(list.Data) != 1 || list.Data[0].ID != "qwen3.8-27b" {
		t.Fatalf("the second listing's models: %+v (%v)", list, err)
	}
}

// The site's per-renter tag reaches the host's gateway as the client a delegating key acts for, so each
// renter is recorded apart and keeps its prompt cache to itself; a malformed tag is dropped on the way.
func TestRentRenterTagReachesTheGatewayAsItsClient(t *testing.T) {
	var mu sync.Mutex
	var seen []string
	f := newRentFixture(t, http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		mu.Lock()
		seen = append(seen, r.Header.Get("X-Emx-Client"))
		mu.Unlock()
		w.Header().Set("Content-Type", "text/event-stream")
		fmt.Fprint(w, "data: [DONE]\n\n")
	}))
	f.startHost(f.hostConfig())
	f.waitAgents(rentTestListing, 1)
	for _, tag := range []string{"a1b2c3d4e5f6a7b8", "", "Bad Tag!"} {
		f.renter = tag
		res := f.forward(context.Background(), rentTestListing, "POST", "/v1/chat/completions", chatBody(rentTestModel, true))
		io.Copy(io.Discard, res.Body)
		res.Body.Close()
	}
	mu.Lock()
	defer mu.Unlock()
	if len(seen) != 3 || seen[0] != "renter-a1b2c3d4e5f6a7b8" || seen[1] != "" || seen[2] != "" {
		t.Fatalf("X-Emx-Client at the target: %q", seen)
	}
}

// A listing hosted with --shared-cache names one cache group for all of its renters.
func TestRentSharedCacheListingNamesOneCacheGroup(t *testing.T) {
	groups := make(chan string, 4)
	f := newRentFixture(t, http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		groups <- r.Header.Get("X-Emx-Cache-Group") + "|" + r.Header.Get("X-Emx-Client")
		w.Header().Set("Content-Type", "text/event-stream")
		fmt.Fprint(w, "data: [DONE]\n\n")
	}))
	cfg := f.hostConfig()
	cfg.SharedCache = []string{rentTestListing}
	f.startHost(cfg)
	f.waitAgents(rentTestListing, 1)
	f.renter = "a1b2c3d4e5f6a7b8"
	res := f.forward(context.Background(), rentTestListing, "POST", "/v1/chat/completions", chatBody(rentTestModel, true))
	io.Copy(io.Discard, res.Body)
	res.Body.Close()
	if got := <-groups; got != "listing-"+rentTestListing+"|renter-a1b2c3d4e5f6a7b8" {
		t.Fatalf("group|client at the target: %q", got)
	}
}

// rentCaptureStdout runs the CLI and returns what it printed to stdout.
func rentCaptureStdout(t *testing.T, argv ...string) string {
	t.Helper()
	original := os.Stdout
	reader, writer, err := os.Pipe()
	if err != nil {
		t.Fatalf("os.Pipe: %v", err)
	}
	os.Stdout = writer
	code := runCLI(argv)
	_ = writer.Close()
	os.Stdout = original
	data, _ := io.ReadAll(reader)
	_ = reader.Close()
	if code != 0 {
		t.Fatalf("runCLI %v exit = %d, output = %s", argv, code, data)
	}
	return string(data)
}

// rent publish sends each model's weights as emx reports them (only once hashed) and their source: emx's
// origin, else --source, which also overrides it.
func TestRentPublishSendsWeightsAndSource(t *testing.T) {
	t.Setenv("HOME", t.TempDir())
	t.Setenv("EMX_API_KEY", "admin-key")
	previousPoll := rentHashingPoll
	rentHashingPoll = 5 * time.Millisecond
	t.Cleanup(func() { rentHashingPoll = previousPoll })

	idA, idB, idC := strings.Repeat("a", 64), strings.Repeat("b", 64), strings.Repeat("c", 64)
	var modelsCalls atomic.Int32
	emx := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if r.URL.Path != "/v1/models" || r.Header.Get("Authorization") != "Bearer admin-key" {
			http.NotFound(w, r)
			return
		}
		// "hashed-late" finishes hashing on the third read; "never-hashed" never does.
		late := `{"status":"hashing"}`
		if modelsCalls.Add(1) >= 3 {
			late = `{"status":"ready","id":"` + idC + `","files":[{"path":"late.gguf","bytes":3,"sha256":"` + idC + `","weight":true}]}`
		}
		price := `"price":{"prompt":1,"completion":2,"currency":"USD"}`
		fmt.Fprint(w, `{"data":[
			{"id":"with-origin","max_model_len":4096,"emx":{`+price+`,"weights":{"status":"ready","id":"`+idA+`","files":[{"path":"Qwen3-0.6B-Q8_0.gguf","bytes":639447744,"sha256":"`+idA+`","weight":true}],"origin":{"repo":"unsloth/Qwen3-0.6B-GGUF","revision":"50968a4468"}}}},
			{"id":"no-origin","emx":{`+price+`,"weights":{"status":"ready","id":"`+idB+`","files":[{"path":"b.gguf","bytes":2,"sha256":"`+idB+`","weight":true}]}}},
			{"id":"hashed-late","emx":{`+price+`,"weights":`+late+`}},
			{"id":"never-hashed","emx":{`+price+`,"weights":{"status":"hashing"}}},
			{"id":"no-report","emx":{`+price+`}}
		]}`)
	}))
	defer emx.Close()
	patches := make(chan []map[string]any, 4)
	site := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if r.Method != http.MethodPatch || r.URL.Path != "/api/rentals/"+rentTestListing || r.Header.Get("Authorization") != "Bearer "+rentTestKey {
			http.Error(w, "unexpected", http.StatusBadRequest)
			return
		}
		var body struct {
			Models []map[string]any `json:"models"`
		}
		_ = json.NewDecoder(r.Body).Decode(&body)
		patches <- body.Models
		fmt.Fprint(w, `{"ok":true}`)
	}))
	defer site.Close()

	publish := func(extra ...string) map[string]map[string]any {
		t.Helper()
		argv := append([]string{"rent", "publish", rentTestListing, "--api-url", site.URL, "--api-key", rentTestKey, "--target", emx.URL}, extra...)
		output := rentCaptureStdout(t, argv...)
		if !strings.Contains(output, "WEIGHTS") {
			t.Fatalf("no weights column:\n%s", output)
		}
		byModel := map[string]map[string]any{}
		for _, m := range <-patches {
			byModel[m["model"].(string)] = m
		}
		return byModel
	}

	got := publish("--model", "with-origin,no-origin,hashed-late,no-report", "--source", "no-origin=Qwen/Qwen3-8B-GGUF@0123456789abcdef")
	weightsID := func(m map[string]any) string {
		w, _ := m["weights"].(map[string]any)
		id, _ := w["id"].(string)
		return id
	}
	source := func(m map[string]any) string {
		s, ok := m["source"].(map[string]any)
		if !ok {
			return ""
		}
		out := s["repo"].(string)
		if rev, ok := s["revision"].(string); ok {
			out += "@" + rev
		}
		return out
	}
	if weightsID(got["with-origin"]) != idA || source(got["with-origin"]) != "unsloth/Qwen3-0.6B-GGUF@50968a4468" {
		t.Fatalf("with-origin = %#v", got["with-origin"])
	}
	if files := got["with-origin"]["weights"].(map[string]any)["files"].([]any); len(files) != 1 || files[0].(map[string]any)["path"] != "Qwen3-0.6B-Q8_0.gguf" {
		t.Fatalf("weights must be published as emx reports them: %#v", got["with-origin"]["weights"])
	}
	if weightsID(got["no-origin"]) != idB || source(got["no-origin"]) != "Qwen/Qwen3-8B-GGUF@0123456789abcdef" {
		t.Fatalf("no-origin with --source = %#v", got["no-origin"])
	}
	if weightsID(got["hashed-late"]) != idC {
		t.Fatalf("publish must wait for hashing to finish: %#v", got["hashed-late"])
	}
	// Without weights ready, weights go out as null, so the site drops any from an earlier publish.
	if w, ok := got["no-report"]["weights"]; !ok || w != nil || source(got["no-report"]) != "" {
		t.Fatalf("no-report = %#v", got["no-report"])
	}

	// --source overrides emx's origin; a model still hashing after --wait-hashing goes out with null weights.
	got = publish("--model", "with-origin,never-hashed", "--source", "with-origin=someone/fork", "--wait-hashing", "20ms")
	if weightsID(got["with-origin"]) != idA || source(got["with-origin"]) != "someone/fork" {
		t.Fatalf("--source override = %#v", got["with-origin"])
	}
	if w, ok := got["never-hashed"]["weights"]; !ok || w != nil {
		t.Fatalf("never-hashed must be published without weights: %#v", got["never-hashed"])
	}

	if code := runCLI([]string{"rent", "publish", rentTestListing, "--api-url", site.URL, "--api-key", rentTestKey, "--target", emx.URL, "--model", "with-origin", "--source", "other=a/b", "--dry-run", "--quiet"}); code == 0 {
		t.Fatal("--source for a model not published must fail")
	}
	if code := runCLI([]string{"rent", "publish", rentTestListing, "--api-url", site.URL, "--api-key", rentTestKey, "--target", emx.URL, "--model", "with-origin", "--source", "with-origin=no-slash", "--dry-run", "--quiet"}); code == 0 {
		t.Fatal("--source without owner/repo must fail")
	}
}

// A listing model published with a weights id is served only while emx reports the same weights for it;
// a listing model without one is served as before.
func TestRentHostSkipsModelWhoseWeightsChanged(t *testing.T) {
	published, live := strings.Repeat("a", 64), atomic.Value{}
	live.Store(strings.Repeat("d", 64))
	emx := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if r.URL.Path != "/v1/models" {
			http.NotFound(w, r)
			return
		}
		fmt.Fprint(w, `{"data":[
			{"id":"same","emx":{"weights":{"status":"ready","id":"`+published+`"}}},
			{"id":"swapped","emx":{"weights":{"status":"ready","id":"`+live.Load().(string)+`"}}},
			{"id":"unhashed","emx":{"weights":{"status":"hashing"}}},
			{"id":"plain","emx":{}}
		]}`)
	}))
	defer emx.Close()
	site := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if r.URL.Path != "/api/rentals/mine" || r.Header.Get("Authorization") != "Bearer "+rentTestKey {
			http.Error(w, "forbidden", http.StatusForbidden)
			return
		}
		fmt.Fprint(w, `{"listings":[{"id":"`+rentTestListing+`","models":[
			{"model":"same","weightsId":"`+published+`","weightsStatus":"VERIFIED"},
			{"model":"swapped","weightsId":"`+published+`","weightsStatus":"VERIFIED"},
			{"model":"unhashed","weightsId":"`+published+`","weightsStatus":"CUSTOM"},
			{"model":"plain","weightsId":null,"weightsStatus":"PENDING"}
		]}]}`)
	}))
	defer site.Close()

	var notices []map[string]any
	cfg := rentHostConfig{Target: emx.URL, APIKey: rentTestKey, SiteURL: site.URL, Listings: []string{rentTestListing},
		Models: []string{"same", "swapped", "unhashed", "plain"}, Client: &http.Client{},
		Status: func(event string, fields map[string]any) {
			if event == "rent_host_weights_changed" {
				notices = append(notices, fields)
			}
		}}
	seen := newRentHostNotices()
	offers, checked, err := rentLoadOffers(cfg, seen)
	if err != nil || !checked {
		t.Fatalf("rentLoadOffers: %v (checked %v)", err, checked)
	}
	cfg.Offers = &rentListingOffers{}
	cfg.Offers.set(offers)
	if got := strings.Join(cfg.Offers.allowed(rentTestListing, cfg.Models), ","); got != "same,plain" {
		t.Fatalf("served = %s, want same,plain", got)
	}
	if len(notices) != 1 || notices[0]["models"] != "swapped, unhashed" || !strings.Contains(notices[0]["note"].(string), "lmx rent publish "+rentTestListing) {
		t.Fatalf("notices = %#v", notices)
	}

	// The next refresh does not repeat the notice; once the weights match again the model is served.
	if _, _, err := rentLoadOffers(cfg, seen); err != nil || len(notices) != 1 {
		t.Fatalf("repeated notice: %#v (%v)", notices, err)
	}
	live.Store(published)
	offers, _, _ = rentLoadOffers(cfg, seen)
	if got := strings.Join(offers[rentTestListing], ","); got != "same,swapped,plain" {
		t.Fatalf("served after the weights match again = %s", got)
	}
}

// The host tells the owner about the site's weights checks once per change of state: nothing while ok, a
// warning at the first failure, a failure notice when the model is out of pools, again after a restore and
// a new failure, and at start when a model is already failed. A failed model is still served directly.
func TestRentHostReportsWeightsChecksOncePerChange(t *testing.T) {
	const reason = "Its answers to recent test requests did not match the majority of 4 hosts checked with these weights (difference 0.51, allowed 0.18; 42% of answers differed early, allowed 25%), so it is most likely not serving the file it reports."
	steps := `["Serve the exact file this model was published with. If you changed the file or its quantization, run ` + "`lmx rent publish " + rentTestListing + "`" + ` again so the listing reports what you serve.","Run it with the engine settings and chat template that are the defaults for that file."]`
	failure := `"lastFailure":{"day":"2026-11-17","score":0.51,"maxScore":0.18,"divergedPercent":42,"maxDivergedPercent":25,"comparedWithHosts":3,"basis":"majority of 4 hosts"}`
	states := map[string]string{
		"ok":        `{"state":"ok","lastCheckedDay":"2026-11-16","failuresInARow":0,"voidedEarningsUsd":0,"reason":"Its answers matched.","whatToDo":[]}`,
		"warning":   `{"state":"warning","lastCheckedDay":"2026-11-17","failuresInARow":1,` + failure + `,"voidedEarningsUsd":0,"reason":"` + reason + `","whatToDo":` + steps + `}`,
		"failed":    `{"state":"failed","lastCheckedDay":"2026-11-17","failuresInARow":2,` + failure + `,"voidedEarningsUsd":12.345,"reason":"` + reason + `","whatToDo":` + steps + `}`,
		"requested": `{"state":"failed","lastCheckedDay":"2026-11-17","failuresInARow":2,` + failure + `,"voidedEarningsUsd":12.345,"reviewRequestedAt":"2026-11-18T09:30:00.000Z","reason":"` + reason + `","whatToDo":` + steps + `}`,
	}
	var current atomic.Value
	current.Store("ok")
	site := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if r.URL.Path != "/api/rentals/mine" || r.Header.Get("Authorization") != "Bearer "+rentTestKey {
			http.Error(w, "forbidden", http.StatusForbidden)
			return
		}
		fmt.Fprint(w, `{"listings":[{"id":"`+rentTestListing+`","models":[{"model":"`+rentTestModel+`","weightsId":null,"weightsCheck":`+states[current.Load().(string)]+`}]}]}`)
	}))
	defer site.Close()

	type notice struct {
		event string
		note  string
	}
	var printed []notice
	cfg := rentHostConfig{APIKey: rentTestKey, SiteURL: site.URL, Listings: []string{rentTestListing}, Models: []string{rentTestModel}, Client: &http.Client{},
		Status: func(event string, fields map[string]any) {
			if strings.HasPrefix(event, "rent_host_check_") {
				if fields["listing"] != rentTestListing || fields["model"] != rentTestModel {
					t.Errorf("%s fields = %#v", event, fields)
				}
				printed = append(printed, notice{event, fields["note"].(string)})
				t.Logf("%s\n%s", event, fields["note"])
			}
		}}
	seen := newRentHostNotices()
	refresh := func(state string, want int) map[string][]string {
		t.Helper()
		current.Store(state)
		served, _, err := rentLoadOffers(cfg, seen)
		if err != nil {
			t.Fatalf("rentLoadOffers (%s): %v", state, err)
		}
		if len(printed) != want {
			t.Fatalf("after %s: %d notices, want %d: %#v", state, len(printed), want, printed)
		}
		return served
	}

	refresh("ok", 0)
	refresh("ok", 0)
	refresh("warning", 1)
	refresh("warning", 1)
	warning := printed[0]
	if warning.event != "rent_host_check_warning" {
		t.Fatalf("first notice = %s", warning.event)
	}
	for _, want := range []string{
		"Failed a hidden weights check (answers compared with other hosts serving the same weights).",
		reason,
		"Failed check of 2026-11-17, judged against the majority of 4 hosts (3 compared); 1 failed in a row.",
		"One more failure removes it from pools and voids its held earnings.",
		"What to do:\n  1. Serve the exact file this model was published with. If you changed the file or its quantization, run lmx rent publish " + rentTestListing + " again",
		"\n  2. Run it with the engine settings and chat template",
		"Details: " + site.URL + "/rentals/" + rentTestListing,
	} {
		if !strings.Contains(warning.note, want) {
			t.Fatalf("warning note lacks %q:\n%s", want, warning.note)
		}
	}

	served := refresh("failed", 2)
	refresh("failed", 2)
	// A review request does not change the state: no new notice.
	refresh("requested", 2)
	failed := printed[1]
	if failed.event != "rent_host_check_failed" {
		t.Fatalf("second notice = %s", failed.event)
	}
	for _, want := range []string{
		"Removed from pools: failed hidden weights checks",
		reason,
		"2 failed in a row.",
		"Out of pools until an admin restores it; held earnings voided: $12.35. Requests sent to this listing directly are still served.",
		"What to do:\n  1. Serve the exact file",
		"Request a review once fixed: " + site.URL + "/rentals/" + rentTestListing,
	} {
		if !strings.Contains(failed.note, want) {
			t.Fatalf("failed note lacks %q:\n%s", want, failed.note)
		}
	}
	if got := strings.Join(served[rentTestListing], ","); got != rentTestModel {
		t.Fatalf("a failed model is still served directly, got %q", got)
	}

	// Restored by an admin (ok, silent), then failing again: warned again.
	refresh("ok", 2)
	refresh("warning", 3)

	// A host starting while the model is failed (review requested) says so at once.
	seen = newRentHostNotices()
	refresh("requested", 4)
	if printed[3].event != "rent_host_check_failed" || !strings.Contains(printed[3].note, "Review requested on 2026-11-18; details: "+site.URL+"/rentals/"+rentTestListing) {
		t.Fatalf("notice at start = %#v", printed[3])
	}
}

// lmx rent list --mine sums up each listing's weights checks in a CHECK column.
func TestRentListMineShowsWeightsChecks(t *testing.T) {
	t.Setenv("HOME", t.TempDir())
	check := func(state string) string { return `{"state":"` + state + `","whatToDo":[]}` }
	site := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if r.URL.Path != "/api/rentals/mine" || r.Header.Get("Authorization") != "Bearer "+rentTestKey {
			http.Error(w, "forbidden", http.StatusForbidden)
			return
		}
		fmt.Fprint(w, `{"listings":[
			{"id":"lst_bad","title":"Bad","status":"ACTIVE","models":[
				{"model":"big","weightsCheck":`+check("failed")+`},{"model":"mid","weightsCheck":`+check("warning")+`},{"model":"small","weightsCheck":`+check("ok")+`}]},
			{"id":"lst_good","title":"Good","status":"ACTIVE","models":[{"model":"small","weightsCheck":`+check("ok")+`},{"model":"custom","weightsCheck":`+check("unchecked")+`}]},
			{"id":"lst_new","title":"New","status":"DRAFT","models":[{"model":"custom","weightsCheck":`+check("unchecked")+`}]}
		]}`)
	}))
	defer site.Close()

	output := rentCaptureStdout(t, "rent", "list", "--mine", "--api-url", site.URL, "--api-key", rentTestKey)
	rows := map[string]string{}
	for _, line := range strings.Split(output, "\n") {
		fields := strings.Fields(strings.ReplaceAll(line, "|", " "))
		if len(fields) > 0 {
			rows[fields[0]] = line
		}
	}
	if !strings.Contains(rows["ID"], "CHECK") {
		t.Fatalf("no CHECK column:\n%s", output)
	}
	for id, want := range map[string]string{"lst_bad": "failed: big; warning: mid", "lst_good": " ok ", "lst_new": " - "} {
		if !strings.Contains(rows[id]+" ", want) {
			t.Fatalf("%s CHECK: want %q in %q\n%s", id, want, rows[id], output)
		}
	}
}
