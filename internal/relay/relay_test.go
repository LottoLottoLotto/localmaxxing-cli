package relay

import (
	"bufio"
	"context"
	"crypto/sha256"
	"encoding/hex"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"net"
	"net/http"
	"net/http/httptest"
	"strings"
	"sync"
	"sync/atomic"
	"testing"
	"time"

	"github.com/LottoLottoLotto/localmaxxing-cli/internal/relayproto"
	"github.com/coder/websocket"
	"github.com/hashicorp/yamux"
)

var (
	testInternalSecret  = strings.Repeat("i", 64)
	testAuthorizeSecret = strings.Repeat("a", 64)
	testKey             = "bhk_" + strings.Repeat("0123456789", 4)
	testKey2            = "bhk_" + strings.Repeat("abcdef0123", 4)
	// testKeySameOwner belongs to testKey's owner (owner_0123).
	testKeySameOwner = "bhk_0123" + strings.Repeat("f", 36)
)

const testListing = "lst_one"

// testEnv is an in-process site authorize endpoint and relay (both listeners).
type testEnv struct {
	t         *testing.T
	srv       *Server
	public    *httptest.Server
	internal  *httptest.Server
	authCalls atomic.Int32
	// authStatus answers authorize after the first call when non-zero.
	authStatus atomic.Int32
	// authHook, when set, runs inside each authorize call.
	authHook atomic.Pointer[func()]
	// agentConcurrency is the hello maxConcurrency (default 4).
	agentConcurrency int
}

func newTestEnv(t *testing.T, configure func(*Config)) *testEnv {
	t.Helper()
	e := &testEnv{t: t}
	site := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if r.URL.Path != "/api/internal/relay/authorize" || r.Header.Get("Authorization") != "Bearer "+testAuthorizeSecret {
			http.Error(w, "forbidden", http.StatusForbidden)
			return
		}
		calls := e.authCalls.Add(1)
		if hook := e.authHook.Load(); hook != nil {
			(*hook)()
		}
		if status := e.authStatus.Load(); status != 0 && calls > 1 {
			w.WriteHeader(int(status))
			return
		}
		var body struct {
			APIKey     string   `json:"apiKey"`
			ListingIDs []string `json:"listingIds"`
		}
		_ = json.NewDecoder(r.Body).Decode(&body)
		listings := []map[string]any{}
		for _, id := range body.ListingIDs {
			listings = append(listings, map[string]any{"id": id, "status": "ACTIVE", "maxConcurrency": 4})
		}
		w.Header().Set("Content-Type", "application/json")
		_ = json.NewEncoder(w).Encode(map[string]any{"ownerId": "owner_" + body.APIKey[4:8], "plan": "PRO", "listings": listings})
	}))
	cfg := Config{SiteURL: site.URL, InternalSecret: testInternalSecret, AuthorizeSecret: testAuthorizeSecret, AuthorizePerMinute: 1000}
	if configure != nil {
		configure(&cfg)
	}
	srv, err := New(cfg)
	if err != nil {
		t.Fatal(err)
	}
	e.srv = srv
	e.public = httptest.NewServer(srv.PublicHandler())
	e.internal = httptest.NewServer(srv.InternalHandler())
	t.Cleanup(func() {
		srv.Close()
		e.public.Close()
		e.internal.Close()
		site.Close()
	})
	return e
}

func (e *testEnv) dial(key, ip string) (*websocket.Conn, *http.Response, error) {
	ctx, cancel := context.WithTimeout(context.Background(), 5*time.Second)
	defer cancel()
	header := http.Header{"Authorization": {"Bearer " + key}}
	if ip != "" {
		header.Set("X-Forwarded-For", ip)
	}
	return websocket.Dial(ctx, "ws"+strings.TrimPrefix(e.public.URL, "http")+relayproto.AgentPath, &websocket.DialOptions{
		Subprotocols: []string{relayproto.Subprotocol}, HTTPHeader: header,
	})
}

// fakeAgent speaks the agent protocol and answers request streams with serve.
type fakeAgent struct {
	session *yamux.Session
	control *relayproto.Conn
}

type controlError struct{ code string }

func (e controlError) Error() string { return e.code }

func (e *testEnv) connect(key, ip string, serve func(req *http.Request, conn net.Conn)) (*fakeAgent, error) {
	ws, _, err := e.dial(key, ip)
	if err != nil {
		return nil, err
	}
	session, err := yamux.Server(relayproto.NetConn(context.Background(), ws), relayproto.YamuxConfig())
	if err != nil {
		return nil, err
	}
	e.t.Cleanup(func() { _ = session.Close() })
	stream, err := session.OpenStream()
	if err != nil {
		return nil, err
	}
	control := relayproto.NewConn(stream)
	concurrency := e.agentConcurrency
	if concurrency == 0 {
		concurrency = 4
	}
	hello := relayproto.Hello{Type: relayproto.TypeHello, Proto: relayproto.Proto, Listings: []string{testListing}, MaxConcurrency: concurrency}
	if err := control.Write(hello, 5*time.Second); err != nil {
		return nil, err
	}
	kind, raw, err := control.Read(5 * time.Second)
	if err != nil {
		return nil, err
	}
	if kind != relayproto.TypeWelcome {
		var notice relayproto.Notice
		_ = json.Unmarshal(raw, &notice)
		return nil, controlError{notice.Code}
	}
	go func() {
		for {
			conn, err := session.AcceptStream()
			if err != nil {
				return
			}
			go func() {
				defer conn.Close()
				req, err := http.ReadRequest(bufio.NewReader(conn))
				if err != nil {
					return
				}
				serve(req, conn)
			}()
		}
	}()
	return &fakeAgent{session: session, control: control}, nil
}

func (e *testEnv) agentCount() int {
	e.srv.mu.Lock()
	defer e.srv.mu.Unlock()
	return len(e.srv.agents)
}

func (e *testEnv) waitAgents(want int) {
	e.t.Helper()
	deadline := time.Now().Add(5 * time.Second)
	for time.Now().Before(deadline) {
		if e.agentCount() == want {
			return
		}
		time.Sleep(10 * time.Millisecond)
	}
	e.t.Fatalf("agents = %d, want %d", e.agentCount(), want)
}

func (e *testEnv) internalRequest(method, path, body string, header http.Header) *http.Response {
	e.t.Helper()
	req, err := http.NewRequest(method, e.internal.URL+path, strings.NewReader(body))
	if err != nil {
		e.t.Fatal(err)
	}
	req.Header.Set("Authorization", "Bearer "+testInternalSecret)
	for name, values := range header {
		req.Header[name] = values
	}
	res, err := http.DefaultClient.Do(req)
	if err != nil {
		e.t.Fatal(err)
	}
	return res
}

func (e *testEnv) forward(method, path string) *http.Response {
	return e.internalRequest(http.MethodPost, "/internal/forward", `{"model":"m"}`, http.Header{
		relayproto.HeaderListing: {testListing}, relayproto.HeaderSession: {"sess_1"},
		relayproto.HeaderMethod: {method}, relayproto.HeaderPath: {path},
	})
}

// rawAgent answers every request with the given raw HTTP response bytes.
func rawAgent(response string) func(*http.Request, net.Conn) {
	return func(_ *http.Request, conn net.Conn) { _, _ = io.WriteString(conn, response) }
}

func relayErrorCode(t *testing.T, res *http.Response) string {
	t.Helper()
	defer res.Body.Close()
	var body relayproto.ErrorBody
	_ = json.NewDecoder(res.Body).Decode(&body)
	if header := res.Header.Get(relayproto.HeaderRelayError); header != body.Error.Code {
		t.Fatalf("%s header %q != body code %q", relayproto.HeaderRelayError, header, body.Error.Code)
	}
	return body.Error.Code
}

func TestNewRefusesWeakSecrets(t *testing.T) {
	for _, secret := range []string{"", "dev", "changeme", "secret", "too-short-0123456789", "replace-with-openssl-rand-hex-32-output-0000000000"} {
		if _, err := New(Config{SiteURL: "http://site", InternalSecret: secret, AuthorizeSecret: testAuthorizeSecret}); err == nil {
			t.Fatalf("internal secret %q accepted", secret)
		}
		if _, err := New(Config{SiteURL: "http://site", InternalSecret: testInternalSecret, AuthorizeSecret: secret}); err == nil {
			t.Fatalf("authorize secret %q accepted", secret)
		}
	}
	if _, err := New(Config{SiteURL: "http://site", InternalSecret: testInternalSecret, AuthorizeSecret: testAuthorizeSecret}); err != nil {
		t.Fatal(err)
	}
}

func TestListenersServeOnlyTheirRoutes(t *testing.T) {
	e := newTestEnv(t, nil)
	get := func(url string) int {
		req, _ := http.NewRequest(http.MethodGet, url, nil)
		req.Header.Set("Authorization", "Bearer "+testInternalSecret)
		res, err := http.DefaultClient.Do(req)
		if err != nil {
			t.Fatal(err)
		}
		res.Body.Close()
		return res.StatusCode
	}
	for _, path := range []string{"/internal/agents?listing=x", "/internal/forward", "/internal/agents/revoke"} {
		if status := get(e.public.URL + path); status != http.StatusNotFound {
			t.Fatalf("public %s = %d, want 404", path, status)
		}
	}
	if status := get(e.internal.URL + "/internal/agents?listing=x"); status != http.StatusOK {
		t.Fatalf("internal agents = %d", status)
	}
	for _, path := range []string{relayproto.AgentPath, "/healthz"} {
		if status := get(e.internal.URL + path); status != http.StatusNotFound {
			t.Fatalf("internal %s = %d, want 404", path, status)
		}
	}
	if status := get(e.public.URL + "/healthz"); status != http.StatusOK {
		t.Fatalf("public healthz = %d", status)
	}
	req, _ := http.NewRequest(http.MethodGet, e.internal.URL+"/internal/agents?listing=x", nil)
	req.Header.Set("Authorization", "Bearer "+testAuthorizeSecret)
	res, err := http.DefaultClient.Do(req)
	if err != nil {
		t.Fatal(err)
	}
	if res.StatusCode != http.StatusUnauthorized || relayErrorCode(t, res) != relayproto.CodeUnauthorized {
		t.Fatalf("wrong internal secret = %d", res.StatusCode)
	}
}

func TestMalformedKeyRejectedBeforeUpgrade(t *testing.T) {
	e := newTestEnv(t, nil)
	for _, key := range []string{"bhk_short", "bhk_" + strings.Repeat("A", 40), testKey + "0", "sk_" + strings.Repeat("0", 40)} {
		_, res, err := e.dial(key, "")
		if err == nil || res == nil || res.StatusCode != http.StatusUnauthorized || res.Header.Get(relayproto.HeaderRelayError) != relayproto.CodeUnauthorized {
			t.Fatalf("key %q: err %v res %+v", key, err, res)
		}
	}
	if calls := e.authCalls.Load(); calls != 0 {
		t.Fatalf("authorize called %d times for malformed keys", calls)
	}
}

func TestResponseTypeNormalizedAndAgentHeadersDropped(t *testing.T) {
	e := newTestEnv(t, nil)
	_, err := e.connect(testKey, "", func(req *http.Request, conn net.Conn) {
		contentType := "text/html"
		if req.URL.Path == "/v1/chat/completions" {
			contentType = "Text/Event-Stream; charset=latin1"
		}
		fmt.Fprintf(conn, "HTTP/1.1 401 Unauthorized\r\nContent-Type: %s\r\n%s: unauthorized\r\nX-Lmx-Other: 1\r\nSet-Cookie: a=b\r\nContent-Length: 2\r\n\r\nok", contentType, relayproto.HeaderRelayError)
	})
	if err != nil {
		t.Fatal(err)
	}
	for path, want := range map[string]string{"/v1/models": "application/json", "/v1/chat/completions": "text/event-stream; charset=utf-8"} {
		method := http.MethodPost
		if path == "/v1/models" {
			method = http.MethodGet
		}
		res := e.forward(method, path)
		body, _ := io.ReadAll(res.Body)
		res.Body.Close()
		if res.StatusCode != http.StatusUnauthorized || string(body) != "ok" || res.Header.Get("Content-Type") != want {
			t.Fatalf("%s: status %d type %q body %q", path, res.StatusCode, res.Header.Get("Content-Type"), body)
		}
		for name := range res.Header {
			if strings.HasPrefix(strings.ToLower(name), "x-lmx-") || name == "Set-Cookie" {
				t.Fatalf("%s: agent header %s reached the site", path, name)
			}
		}
	}
	res := e.internalRequest(http.MethodPost, "/internal/forward", "{}", http.Header{
		relayproto.HeaderListing: {"lst_nobody"}, relayproto.HeaderMethod: {"POST"}, relayproto.HeaderPath: {"/v1/completions"},
	})
	if res.StatusCode != http.StatusServiceUnavailable || relayErrorCode(t, res) != "no_agent" {
		t.Fatalf("no agent = %d", res.StatusCode)
	}
}

func TestOversizedResponseHeadersRejected(t *testing.T) {
	e := newTestEnv(t, nil)
	_, err := e.connect(testKey, "", func(req *http.Request, conn net.Conn) {
		if req.URL.Path == "/v1/models" {
			fmt.Fprintf(conn, "HTTP/1.1 200 OK\r\nX-Big: %s\r\n\r\n", strings.Repeat("a", 70<<10))
			return
		}
		var fields strings.Builder
		for i := range maxResponseHeaderFields + 1 {
			fmt.Fprintf(&fields, "X-F%d: v\r\n", i)
		}
		fmt.Fprintf(conn, "HTTP/1.1 200 OK\r\n%sContent-Length: 0\r\n\r\n", fields.String())
	})
	if err != nil {
		t.Fatal(err)
	}
	for _, route := range [][2]string{{"GET", "/v1/models"}, {"POST", "/v1/embeddings"}} {
		res := e.forward(route[0], route[1])
		if res.StatusCode != http.StatusBadGateway || relayErrorCode(t, res) != "agent_failed" {
			t.Fatalf("%s: status %d, want 502 agent_failed", route[1], res.StatusCode)
		}
	}
}

func TestOversizedResponseBodyCapped(t *testing.T) {
	e := newTestEnv(t, func(cfg *Config) {
		cfg.MaxResponseBytes = 1024
		cfg.MaxStreamResponseBytes = 4096
	})
	_, err := e.connect(testKey, "", func(req *http.Request, conn net.Conn) {
		switch req.URL.Path {
		case "/v1/models": // declared length over the cap
			fmt.Fprintf(conn, "HTTP/1.1 200 OK\r\nContent-Type: application/json\r\nContent-Length: 2048\r\n\r\n%s", strings.Repeat("a", 2048))
		case "/v1/embeddings": // chunked, over the cap
			fmt.Fprintf(conn, "HTTP/1.1 200 OK\r\nContent-Type: application/json\r\nTransfer-Encoding: chunked\r\n\r\n800\r\n%s\r\n0\r\n\r\n", strings.Repeat("a", 2048))
		default: // SSE: 2048 bytes is under the stream cap
			fmt.Fprintf(conn, "HTTP/1.1 200 OK\r\nContent-Type: text/event-stream\r\nTransfer-Encoding: chunked\r\n\r\n800\r\n%s\r\n0\r\n\r\n", strings.Repeat("a", 2048))
		}
	})
	if err != nil {
		t.Fatal(err)
	}
	res := e.forward("GET", "/v1/models")
	if res.StatusCode != http.StatusBadGateway || relayErrorCode(t, res) != "response_too_large" {
		t.Fatalf("declared oversize: status %d", res.StatusCode)
	}
	res = e.forward("POST", "/v1/embeddings")
	body, err := io.ReadAll(res.Body)
	res.Body.Close()
	if err == nil || len(body) > 1024 {
		t.Fatalf("chunked oversize: read %d bytes, err %v; want an aborted response", len(body), err)
	}
	res = e.forward("POST", "/v1/chat/completions")
	body, err = io.ReadAll(res.Body)
	res.Body.Close()
	if err != nil || len(body) != 2048 {
		t.Fatalf("SSE under cap: read %d bytes, err %v", len(body), err)
	}
}

func TestAgentOpenedStreamDisconnects(t *testing.T) {
	e := newTestEnv(t, nil)
	agent, err := e.connect(testKey, "", rawAgent(""))
	if err != nil {
		t.Fatal(err)
	}
	e.waitAgents(1)
	extra, err := agent.session.OpenStream()
	if err != nil {
		t.Fatal(err)
	}
	_, _ = extra.Write(make([]byte, 64<<10))
	select {
	case <-agent.session.CloseChan():
	case <-time.After(5 * time.Second):
		t.Fatal("relay kept an agent that opened an extra stream")
	}
	e.waitAgents(0)
}

func TestAgentConnectionLimits(t *testing.T) {
	t.Run("per IP, handshakes included", func(t *testing.T) {
		e := newTestEnv(t, func(cfg *Config) { cfg.MaxAgentsPerIP = 2 })
		for range 2 {
			ws, _, err := e.dial(testKey, "203.0.113.5")
			if err != nil {
				t.Fatal(err)
			}
			defer ws.CloseNow()
		}
		if _, res, err := e.dial(testKey, "203.0.113.5"); err == nil || res == nil || res.StatusCode != http.StatusTooManyRequests || res.Header.Get(relayproto.HeaderRelayError) != relayproto.CodeTooManyAgents {
			t.Fatalf("third connection from one IP: err %v res %v", err, res)
		}
		// A client-supplied left XFF entry does not change the address.
		if _, res, err := e.dial(testKey, "198.51.100.1, 203.0.113.5"); err == nil || res.StatusCode != http.StatusTooManyRequests {
			t.Fatalf("spoofed XFF bypassed the per-IP cap: %v", err)
		}
		if _, err := e.connect(testKey, "203.0.113.6", rawAgent("")); err != nil {
			t.Fatalf("other IP: %v", err)
		}
	})
	t.Run("handshakes separate from registered agents", func(t *testing.T) {
		e := newTestEnv(t, func(cfg *Config) { cfg.MaxHandshakes = 1 })
		ws, _, err := e.dial(testKey, "203.0.113.5")
		if err != nil {
			t.Fatal(err)
		}
		if _, res, err := e.dial(testKey, "203.0.113.6"); err == nil || res == nil || res.StatusCode != http.StatusTooManyRequests || res.Header.Get(relayproto.HeaderRelayError) != relayproto.CodeTooManyAgents {
			t.Fatalf("second handshake: err %v res %v", err, res)
		}
		ws.CloseNow()
		e.waitAdmission(func() bool { return e.srv.handshakes == 0 })
		// Registered agents free their handshake slot.
		for _, key := range []string{testKey, testKey2} {
			if _, err := e.connect(key, "203.0.113.7", rawAgent("")); err != nil {
				t.Fatalf("registered agent %s: %v", key[:8], err)
			}
		}
		e.waitAgents(2)
	})
	t.Run("registered total", func(t *testing.T) {
		e := newTestEnv(t, func(cfg *Config) { cfg.MaxAgents = 1 })
		// Handshakes do not count against MaxAgents.
		ws, _, err := e.dial(testKey, "203.0.113.5")
		if err != nil {
			t.Fatal(err)
		}
		defer ws.CloseNow()
		if _, err := e.connect(testKey, "203.0.113.6", rawAgent("")); err != nil {
			t.Fatal(err)
		}
		if _, res, err := e.dial(testKey2, "203.0.113.7"); err == nil || res == nil || res.StatusCode != http.StatusTooManyRequests {
			t.Fatalf("over registered cap: err %v", err)
		}
	})
	t.Run("per owner", func(t *testing.T) {
		e := newTestEnv(t, func(cfg *Config) { cfg.MaxAgentsPerOwner = 1 })
		if _, err := e.connect(testKey, "203.0.113.5", rawAgent("")); err != nil {
			t.Fatal(err)
		}
		var refused controlError
		if _, err := e.connect(testKeySameOwner, "203.0.113.6", rawAgent("")); !errors.As(err, &refused) || refused.code != relayproto.CodeTooManyAgents {
			t.Fatalf("second agent of one owner: %v", err)
		}
		if _, err := e.connect(testKey2, "203.0.113.7", rawAgent("")); err != nil {
			t.Fatalf("other owner: %v", err)
		}
	})
	t.Run("IPv6 per /64", func(t *testing.T) {
		e := newTestEnv(t, func(cfg *Config) { cfg.MaxAgentsPerIP = 1 })
		ws, _, err := e.dial(testKey, "2001:db8::1")
		if err != nil {
			t.Fatal(err)
		}
		defer ws.CloseNow()
		if _, res, err := e.dial(testKey, "2001:db8::2"); err == nil || res == nil || res.StatusCode != http.StatusTooManyRequests {
			t.Fatalf("same /64: err %v", err)
		}
		if _, err := e.connect(testKey, "2001:db8:0:1::1", rawAgent("")); err != nil {
			t.Fatalf("other /64: %v", err)
		}
	})
	t.Run("per API key", func(t *testing.T) {
		e := newTestEnv(t, func(cfg *Config) { cfg.MaxAgentsPerKey = 1 })
		if _, err := e.connect(testKey, "203.0.113.5", rawAgent("")); err != nil {
			t.Fatal(err)
		}
		var refused controlError
		if _, err := e.connect(testKey, "203.0.113.6", rawAgent("")); !errors.As(err, &refused) || refused.code != relayproto.CodeTooManyAgents {
			t.Fatalf("second agent on one key: %v", err)
		}
		if _, err := e.connect(testKey2, "203.0.113.7", rawAgent("")); err != nil {
			t.Fatalf("other key: %v", err)
		}
	})
	t.Run("connection attempts per IP", func(t *testing.T) {
		e := newTestEnv(t, func(cfg *Config) { cfg.AuthorizePerMinute = 2 })
		for range 2 {
			ws, _, err := e.dial(testKey, "203.0.113.5")
			if err != nil {
				t.Fatal(err)
			}
			ws.CloseNow()
		}
		if _, res, err := e.dial(testKey, "203.0.113.5"); err == nil || res == nil || res.StatusCode != http.StatusTooManyRequests || res.Header.Get(relayproto.HeaderRelayError) != relayproto.CodeRateLimited {
			t.Fatalf("third attempt in a minute: err %v", err)
		}
	})
	t.Run("full address table evicts the least recently used", func(t *testing.T) {
		e := newTestEnv(t, func(cfg *Config) { cfg.AuthorizePerMinute = 1 })
		e.srv.admitMu.Lock()
		e.srv.maxBuckets = 2
		e.srv.admitMu.Unlock()
		attempt := func(ip string) int {
			ws, res, err := e.dial(testKey, ip)
			if err == nil {
				ws.CloseNow()
				return http.StatusSwitchingProtocols
			}
			if res == nil {
				t.Fatalf("%s: %v", ip, err)
			}
			return res.StatusCode
		}
		for _, ip := range []string{"203.0.113.1", "203.0.113.2", "203.0.113.3"} {
			if status := attempt(ip); status != http.StatusSwitchingProtocols {
				t.Fatalf("new address %s = %d", ip, status)
			}
		}
		// .1 was evicted for .3: it starts with a fresh bucket (evicting .2).
		if status := attempt("203.0.113.1"); status != http.StatusSwitchingProtocols {
			t.Fatalf("evicted address = %d, want a fresh bucket", status)
		}
		// .3 is still tracked with its token spent.
		if status := attempt("203.0.113.3"); status != http.StatusTooManyRequests {
			t.Fatalf("tracked address = %d, want 429", status)
		}
		e.srv.admitMu.Lock()
		tracked := len(e.srv.buckets)
		e.srv.admitMu.Unlock()
		if tracked != 2 {
			t.Fatalf("tracked addresses = %d, want 2", tracked)
		}
	})
	t.Run("handshakes per address", func(t *testing.T) {
		e := newTestEnv(t, func(cfg *Config) { cfg.MaxHandshakesPerIP = 1 })
		ws, _, err := e.dial(testKey, "2001:db8::1")
		if err != nil {
			t.Fatal(err)
		}
		if _, res, err := e.dial(testKey, "2001:db8::2"); err == nil || res == nil || res.StatusCode != http.StatusTooManyRequests || res.Header.Get(relayproto.HeaderRelayError) != relayproto.CodeTooManyAgents {
			t.Fatalf("second handshake from one /64: err %v", err)
		}
		if _, err := e.connect(testKey, "203.0.113.9", rawAgent("")); err != nil {
			t.Fatalf("other address: %v", err)
		}
		ws.CloseNow()
		e.waitAdmission(func() bool { return e.srv.handshakes == 0 })
		// Registered agents do not hold handshake slots.
		for _, key := range []string{testKey2, testKeySameOwner} {
			if _, err := e.connect(key, "2001:db8::3", rawAgent("")); err != nil {
				t.Fatalf("sequential agent from the /64: %v", err)
			}
		}
	})
}

func TestHandshakeHasOneDeadline(t *testing.T) {
	const timeout = time.Second
	e := newTestEnv(t, func(cfg *Config) { cfg.HandshakeTimeout = timeout })
	started := time.Now()
	ws, _, err := e.dial(testKey, "")
	if err != nil {
		t.Fatal(err)
	}
	session, err := yamux.Server(relayproto.NetConn(context.Background(), ws), relayproto.YamuxConfig())
	if err != nil {
		t.Fatal(err)
	}
	defer session.Close()
	// Open the control stream late and never send hello.
	time.Sleep(timeout * 8 / 10)
	if _, err := session.OpenStream(); err != nil {
		t.Fatal(err)
	}
	select {
	case <-session.CloseChan():
	case <-time.After(5 * time.Second):
		t.Fatal("relay kept a connection that never sent hello")
	}
	// Separate accept and hello timeouts would allow 1.8s.
	if elapsed := time.Since(started); elapsed > timeout*3/2 {
		t.Fatalf("handshake lasted %v, want about %v", elapsed, timeout)
	}
	e.waitAdmission(func() bool { return e.srv.handshakes == 0 })
}

// waitAdmission waits for cond, evaluated under the admission lock.
func (e *testEnv) waitAdmission(cond func() bool) {
	e.t.Helper()
	deadline := time.Now().Add(5 * time.Second)
	for time.Now().Before(deadline) {
		e.srv.admitMu.Lock()
		ok := cond()
		e.srv.admitMu.Unlock()
		if ok {
			return
		}
		time.Sleep(10 * time.Millisecond)
	}
	e.t.Fatal("admission state did not settle")
}

func TestNonUpgradeRequestsSkipAdmission(t *testing.T) {
	e := newTestEnv(t, nil)
	upgrade := http.Header{
		"Authorization": {"Bearer " + testKey}, "Connection": {"Upgrade"}, "Upgrade": {"websocket"},
		"Sec-Websocket-Version": {"13"}, "Sec-Websocket-Key": {"dGhlIHNhbXBsZSBub25jZQ=="}, "Sec-Websocket-Protocol": {relayproto.Subprotocol},
	}
	without := func(name string) http.Header {
		header := upgrade.Clone()
		header.Del(name)
		return header
	}
	badKey := upgrade.Clone()
	badKey.Set("Sec-Websocket-Key", "short")
	cases := map[string]struct {
		method string
		header http.Header
	}{
		"plain GET":        {http.MethodGet, http.Header{"Authorization": {"Bearer " + testKey}}},
		"POST":             {http.MethodPost, upgrade},
		"no Upgrade":       {http.MethodGet, without("Upgrade")},
		"no Connection":    {http.MethodGet, without("Connection")},
		"no version":       {http.MethodGet, without("Sec-Websocket-Version")},
		"no key":           {http.MethodGet, without("Sec-Websocket-Key")},
		"bad key":          {http.MethodGet, badKey},
		"no subprotocol":   {http.MethodGet, without("Sec-Websocket-Protocol")},
		"bad API key":      {http.MethodGet, http.Header{"Authorization": {"Bearer bhk_x"}}},
		"no Authorization": {http.MethodGet, http.Header{}},
	}
	for name, c := range cases {
		req, _ := http.NewRequest(c.method, e.public.URL+relayproto.AgentPath, nil)
		req.Header = c.header
		res, err := http.DefaultClient.Do(req)
		if err != nil {
			t.Fatal(err)
		}
		res.Body.Close()
		if res.StatusCode != http.StatusUpgradeRequired && res.StatusCode != http.StatusUnauthorized {
			t.Fatalf("%s: status %d", name, res.StatusCode)
		}
	}
	e.srv.admitMu.Lock()
	buckets, handshakes := len(e.srv.buckets), e.srv.handshakes
	e.srv.admitMu.Unlock()
	if buckets != 0 || handshakes != 0 || e.authCalls.Load() != 0 {
		t.Fatalf("buckets %d handshakes %d authorize calls %d, want none", buckets, handshakes, e.authCalls.Load())
	}
}

func TestClientIP(t *testing.T) {
	cases := []struct{ remote, xff, cf, header, want string }{
		{"198.51.100.7:1234", "203.0.113.1", "", "", "198.51.100.7"},       // public peer: XFF ignored
		{"172.18.0.3:1234", "1.1.1.1, 203.0.113.9", "", "", "203.0.113.9"}, // proxy: rightmost entry
		{"127.0.0.1:1234", "", "", "", "127.0.0.1"},
		{"10.0.0.2:1234", "not-an-ip", "", "", "10.0.0.2"},
		// Configured header (CDN in front of the proxy).
		{"172.18.0.3:1234", "198.51.100.1, 104.16.0.1", "203.0.113.4", "CF-Connecting-IP", "203.0.113.4"},
		{"172.18.0.3:1234", "104.16.0.1", "2001:db8::5", "CF-Connecting-IP", "2001:db8::5"},
		{"172.18.0.3:1234", "104.16.0.1", "", "CF-Connecting-IP", "104.16.0.1"},      // header missing: XFF
		{"172.18.0.3:1234", "104.16.0.1", "junk", "CF-Connecting-IP", "104.16.0.1"},  // header invalid: XFF
		{"198.51.100.7:1234", "", "203.0.113.4", "CF-Connecting-IP", "198.51.100.7"}, // public peer: header ignored
		{"172.18.0.3:1234", "104.16.0.1", "203.0.113.4", "", "104.16.0.1"},           // not configured: header ignored
	}
	for _, c := range cases {
		r := httptest.NewRequest(http.MethodGet, "/", nil)
		r.RemoteAddr = c.remote
		if c.xff != "" {
			r.Header.Set("X-Forwarded-For", c.xff)
		}
		if c.cf != "" {
			r.Header.Set("CF-Connecting-IP", c.cf)
		}
		if got := clientIP(r, c.header); got != c.want {
			t.Fatalf("%s + %q + cf %q (header %q) = %s, want %s", c.remote, c.xff, c.cf, c.header, got, c.want)
		}
	}
	for ip, want := range map[string]string{
		"203.0.113.4": "203.0.113.4", "2001:db8::5": "2001:db8::/64", "2001:db8:0:0:ffff::1": "2001:db8::/64",
		"2001:db8:0:1::1": "2001:db8:0:1::/64", "pipe": "pipe",
	} {
		if got := addressKey(ip); got != want {
			t.Fatalf("addressKey(%s) = %s, want %s", ip, got, want)
		}
	}
}

func TestClientIPHeaderConfig(t *testing.T) {
	if _, err := New(Config{SiteURL: "http://site", InternalSecret: testInternalSecret, AuthorizeSecret: testAuthorizeSecret, ClientIPHeader: "CF Connecting IP"}); err == nil {
		t.Fatal("invalid header name accepted")
	}
	e := newTestEnv(t, func(cfg *Config) {
		cfg.ClientIPHeader = "cf-connecting-ip"
		cfg.MaxAgentsPerIP = 1
	})
	dial := func(client string) (*websocket.Conn, *http.Response, error) {
		ctx, cancel := context.WithTimeout(context.Background(), 5*time.Second)
		defer cancel()
		return websocket.Dial(ctx, "ws"+strings.TrimPrefix(e.public.URL, "http")+relayproto.AgentPath, &websocket.DialOptions{
			Subprotocols: []string{relayproto.Subprotocol},
			HTTPHeader:   http.Header{"Authorization": {"Bearer " + testKey}, "X-Forwarded-For": {"104.16.0.1"}, "Cf-Connecting-Ip": {client}},
		})
	}
	ws, _, err := dial("203.0.113.4")
	if err != nil {
		t.Fatal(err)
	}
	defer ws.CloseNow()
	// Same CDN edge (XFF), different client: its own limit.
	other, _, err := dial("203.0.113.5")
	if err != nil {
		t.Fatalf("second client behind the same edge: %v", err)
	}
	defer other.CloseNow()
	if _, res, err := dial("203.0.113.4"); err == nil || res == nil || res.StatusCode != http.StatusTooManyRequests {
		t.Fatalf("same client twice: err %v", err)
	}
}

// relayStreams is the relay-side stream count of the only agent's session
// (the control stream included).
func (e *testEnv) relayStreams() int {
	agents := e.srv.snapshot()
	if len(agents) != 1 {
		e.t.Fatalf("agents = %d, want 1", len(agents))
	}
	return agents[0].session.NumStreams()
}

func TestBytesAfterResponseDisconnectAgent(t *testing.T) {
	for name, serve := range map[string]func(*http.Request, net.Conn){
		"buffered with the body": rawAgent("HTTP/1.1 200 OK\r\nContent-Type: application/json\r\nContent-Length: 2\r\n\r\nokJUNK" + strings.Repeat("j", 64<<10)),
		"after the body": func(_ *http.Request, conn net.Conn) {
			_, _ = io.WriteString(conn, "HTTP/1.1 200 OK\r\nContent-Type: application/json\r\nContent-Length: 2\r\n\r\nok")
			time.Sleep(200 * time.Millisecond)
			_, _ = conn.Write(make([]byte, 64<<10))
			time.Sleep(time.Second)
		},
	} {
		t.Run(name, func(t *testing.T) {
			e := newTestEnv(t, nil)
			agent, err := e.connect(testKey, "", serve)
			if err != nil {
				t.Fatal(err)
			}
			res := e.forward(http.MethodGet, "/v1/models")
			body, err := io.ReadAll(res.Body)
			res.Body.Close()
			if err != nil || res.StatusCode != http.StatusOK || string(body) != "ok" {
				t.Fatalf("status %d body %q err %v", res.StatusCode, body, err)
			}
			select {
			case <-agent.session.CloseChan():
			case <-time.After(5 * time.Second):
				t.Fatal("relay kept an agent that sent bytes after its response")
			}
			e.waitAgents(0)
		})
	}
	t.Run("clean close keeps the agent", func(t *testing.T) {
		e := newTestEnv(t, nil)
		if _, err := e.connect(testKey, "", rawAgent("HTTP/1.1 200 OK\r\nContent-Length: 2\r\n\r\nok")); err != nil {
			t.Fatal(err)
		}
		for range 3 {
			res := e.forward(http.MethodGet, "/v1/models")
			_, _ = io.Copy(io.Discard, res.Body)
			res.Body.Close()
			if res.StatusCode != http.StatusOK {
				t.Fatalf("status %d", res.StatusCode)
			}
		}
		deadline := time.Now().Add(5 * time.Second)
		for e.relayStreams() != 1 && time.Now().Before(deadline) {
			time.Sleep(10 * time.Millisecond)
		}
		if streams := e.relayStreams(); streams != 1 {
			t.Fatalf("relay streams = %d, want only the control stream", streams)
		}
	})
}

func TestRenterCancelFreesStream(t *testing.T) {
	const closeTimeout = 300 * time.Millisecond
	e := newTestEnv(t, func(cfg *Config) { cfg.StreamCloseTimeout = closeTimeout })
	cancelled, done := make(chan struct{}), make(chan struct{})
	t.Cleanup(func() { close(done) })
	_, err := e.connect(testKey, "", func(_ *http.Request, conn net.Conn) {
		_, _ = io.WriteString(conn, "HTTP/1.1 200 OK\r\nContent-Type: text/event-stream\r\nTransfer-Encoding: chunked\r\n\r\n5\r\nhello\r\n")
		<-cancelled
		// Keep sending without ever closing: the relay must not hold this.
		_, _ = fmt.Fprintf(conn, "%x\r\n", 200<<10)
		_, _ = conn.Write(make([]byte, 200<<10))
		<-done
	})
	if err != nil {
		t.Fatal(err)
	}
	ctx, cancel := context.WithCancel(context.Background())
	req, _ := http.NewRequestWithContext(ctx, http.MethodPost, e.internal.URL+"/internal/forward", strings.NewReader(`{"model":"m"}`))
	req.Header = http.Header{
		"Authorization": {"Bearer " + testInternalSecret}, relayproto.HeaderListing: {testListing},
		relayproto.HeaderMethod: {http.MethodPost}, relayproto.HeaderPath: {"/v1/chat/completions"},
	}
	res, err := http.DefaultClient.Do(req)
	if err != nil {
		t.Fatal(err)
	}
	first := make([]byte, 5)
	if _, err := io.ReadFull(res.Body, first); err != nil || string(first) != "hello" {
		t.Fatalf("first chunk %q (%v)", first, err)
	}
	if streams := e.relayStreams(); streams != 2 {
		t.Fatalf("relay streams during the request = %d, want 2", streams)
	}
	cancel()
	res.Body.Close()
	close(cancelled)
	deadline := time.Now().Add(closeTimeout + 2*time.Second)
	for e.relayStreams() != 1 {
		if time.Now().After(deadline) {
			t.Fatalf("relay streams = %d after the close timeout, want only the control stream", e.relayStreams())
		}
		time.Sleep(10 * time.Millisecond)
	}
}

func TestPickHonorsMaxConcurrency(t *testing.T) {
	e := newTestEnv(t, nil)
	e.agentConcurrency = 1
	release := make(chan struct{})
	t.Cleanup(func() { close(release) })
	serve := func(req *http.Request, conn net.Conn) {
		if req.URL.Path == "/v1/chat/completions" {
			<-release
		}
		_, _ = io.WriteString(conn, "HTTP/1.1 200 OK\r\nContent-Length: 2\r\n\r\nok")
	}
	if _, err := e.connect(testKey, "", serve); err != nil {
		t.Fatal(err)
	}
	go func() {
		res := e.forward(http.MethodPost, "/v1/chat/completions")
		res.Body.Close()
	}()
	deadline := time.Now().Add(5 * time.Second)
	for e.srv.snapshot()[0].inFlight.Load() != 1 {
		if time.Now().After(deadline) {
			t.Fatal("first request never reached the agent")
		}
		time.Sleep(10 * time.Millisecond)
	}
	if res := e.forward(http.MethodGet, "/v1/models"); res.StatusCode != http.StatusServiceUnavailable || relayErrorCode(t, res) != "no_agent" {
		t.Fatalf("request past maxConcurrency = %d, want 503 no_agent", res.StatusCode)
	}
	if _, err := e.connect(testKey2, "", serve); err != nil {
		t.Fatal(err)
	}
	res := e.forward(http.MethodGet, "/v1/models")
	res.Body.Close()
	if res.StatusCode != http.StatusOK {
		t.Fatalf("request with a free agent = %d", res.StatusCode)
	}
}

func TestAgentFailureMidBodyAbortsResponse(t *testing.T) {
	e := newTestEnv(t, nil)
	_, err := e.connect(testKey, "", func(req *http.Request, conn net.Conn) {
		if req.URL.Path == "/v1/chat/completions" { // chunked, then the stream ends
			_, _ = io.WriteString(conn, "HTTP/1.1 200 OK\r\nContent-Type: text/event-stream\r\nTransfer-Encoding: chunked\r\n\r\n5\r\nhello\r\n")
			return
		}
		// Length declared, body cut short.
		_, _ = io.WriteString(conn, "HTTP/1.1 200 OK\r\nContent-Type: application/json\r\nContent-Length: 100\r\n\r\n{\"partial\":")
	})
	if err != nil {
		t.Fatal(err)
	}
	for _, route := range [][2]string{{"POST", "/v1/chat/completions"}, {"POST", "/v1/embeddings"}} {
		res := e.forward(route[0], route[1])
		body, err := io.ReadAll(res.Body)
		res.Body.Close()
		if err == nil {
			t.Fatalf("%s: read %q with a clean end, want a failed transfer", route[1], body)
		}
	}
}

func TestRegisterRefusesKeysRevokedDuringAuthorize(t *testing.T) {
	keySum := sha256.Sum256([]byte(testKey))
	for name, revocation := range map[string]string{
		"api key": `{"apiKeyHash":"` + hex.EncodeToString(keySum[:]) + `"}`,
		"owner":   `{"ownerId":"owner_0123"}`,
	} {
		t.Run(name, func(t *testing.T) {
			e := newTestEnv(t, nil)
			entered, release := make(chan struct{}), make(chan struct{})
			var once sync.Once
			hook := func() {
				once.Do(func() {
					close(entered)
					<-release
				})
			}
			e.authHook.Store(&hook)
			result := make(chan error, 1)
			go func() {
				_, err := e.connect(testKey, "", rawAgent(""))
				result <- err
			}()
			<-entered
			res := e.internalRequest(http.MethodPost, "/internal/agents/revoke", revocation, nil)
			res.Body.Close()
			close(release)
			var refused controlError
			if err := <-result; !errors.As(err, &refused) || refused.code != relayproto.CodeRevoked {
				t.Fatalf("agent authorized before the revoke registered after it: %v", err)
			}
			if !relayproto.Terminal(relayproto.CodeRevoked) {
				t.Fatal("revoked must stop the agent")
			}
			// Authorized after the revocation (the site decides): accepted.
			if _, err := e.connect(testKey, "", rawAgent("")); err != nil {
				t.Fatalf("agent authorized after the revoke: %v", err)
			}
		})
	}
}

func TestReauthorizeFailsClosedAfterThreeFailures(t *testing.T) {
	e := newTestEnv(t, func(cfg *Config) { cfg.Reauthorize = 30 * time.Millisecond })
	e.authStatus.Store(http.StatusInternalServerError)
	agent, err := e.connect(testKey, "", rawAgent(""))
	if err != nil {
		t.Fatal(err)
	}
	kind, raw, err := agent.control.Read(5 * time.Second)
	if err != nil || kind != relayproto.TypeRevoked {
		t.Fatalf("control message %q (%v), want revoked", kind, err)
	}
	var notice relayproto.Notice
	_ = json.Unmarshal(raw, &notice)
	if notice.Code != relayproto.CodeSiteUnavailable {
		t.Fatalf("revoke code %q", notice.Code)
	}
	if calls := e.authCalls.Load(); calls != 1+maxReauthorizeFailures {
		t.Fatalf("authorize calls = %d, want %d", calls, 1+maxReauthorizeFailures)
	}
	e.waitAgents(0)
}

func TestRevokeByAPIKeyHash(t *testing.T) {
	e := newTestEnv(t, nil)
	first, err := e.connect(testKey, "", rawAgent(""))
	if err != nil {
		t.Fatal(err)
	}
	if _, err := e.connect(testKey2, "", rawAgent("")); err != nil {
		t.Fatal(err)
	}
	e.waitAgents(2)
	sum := sha256.Sum256([]byte(testKey))
	res := e.internalRequest(http.MethodPost, "/internal/agents/revoke", `{"apiKeyHash":"`+hex.EncodeToString(sum[:])+`"}`, nil)
	var body struct {
		Revoked int `json:"revoked"`
	}
	_ = json.NewDecoder(res.Body).Decode(&body)
	res.Body.Close()
	if body.Revoked != 1 {
		t.Fatalf("revoked = %d, want 1", body.Revoked)
	}
	kind, raw, err := first.control.Read(5 * time.Second)
	var notice relayproto.Notice
	_ = json.Unmarshal(raw, &notice)
	if err != nil || kind != relayproto.TypeRevoked || notice.Code != relayproto.CodeUnauthorized {
		t.Fatalf("revoked agent got %q %q (%v)", kind, notice.Code, err)
	}
	e.waitAgents(1)
	for _, bad := range []string{`{}`, `{"ownerId":"o","apiKeyHash":"` + hex.EncodeToString(sum[:]) + `"}`, `{"apiKeyHash":"` + testKey + `"}`} {
		if res := e.internalRequest(http.MethodPost, "/internal/agents/revoke", bad, nil); res.StatusCode != http.StatusBadRequest || relayErrorCode(t, res) != "bad_request" {
			t.Fatalf("revoke %s = %d", bad, res.StatusCode)
		}
	}
}
