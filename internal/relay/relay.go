// Package relay is the lmx-relay transport: owner agents (`lmx rent host`)
// hold an outbound WebSocket+yamux connection, and the site forwards renter
// requests through /internal/forward to an agent serving the listing. The
// relay has no database: the site authorizes agents and owns access,
// capacity and billing. Agents (owner machines) are untrusted.
package relay

import (
	"bufio"
	"bytes"
	"container/list"
	"context"
	"crypto/rand"
	"crypto/sha256"
	"crypto/subtle"
	"encoding/base64"
	"encoding/hex"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"log/slog"
	"net"
	"net/http"
	"net/netip"
	"regexp"
	"strconv"
	"strings"
	"sync"
	"sync/atomic"
	"time"

	"github.com/LottoLottoLotto/localmaxxing-cli/internal/relayproto"
	"github.com/coder/websocket"
	"github.com/hashicorp/yamux"
)

// Defaults for the Config limits.
const (
	DefaultMaxHandshakes          = 256
	DefaultMaxHandshakesPerIP     = 2
	DefaultHandshakeTimeout       = 5 * time.Second
	DefaultMaxAgents              = 2000
	DefaultMaxAgentsPerIP         = 20
	DefaultMaxAgentsPerKey        = 8
	DefaultMaxAgentsPerOwner      = 32
	DefaultAuthorizePerMinute     = 10
	DefaultMaxResponseBytes       = 32 << 20
	DefaultMaxStreamResponseBytes = 256 << 20
	DefaultStreamCloseTimeout     = 5 * time.Second
	// MinSecretLength is the shortest accepted RELAY_*_SECRET.
	MinSecretLength = 32
)

// Config configures a relay Server. Zero durations and limits take the defaults.
type Config struct {
	// SiteURL is the site origin used for /api/internal/relay/authorize.
	SiteURL string
	// InternalSecret (RELAY_INTERNAL_SECRET) authenticates the site's calls to /internal/*.
	InternalSecret string
	// AuthorizeSecret (RELAY_AUTHORIZE_SECRET) authenticates the relay's calls to the site.
	AuthorizeSecret string
	// HTTPClient calls the site (default: 10s timeout).
	HTTPClient *http.Client
	// Heartbeat is the control ping interval (default 20s).
	Heartbeat time.Duration
	// Reauthorize is how often each agent is re-authorized (default 5m).
	Reauthorize time.Duration
	// ResponseHeaderTimeout bounds waiting for an agent's response headers (default 120s).
	ResponseHeaderTimeout time.Duration
	// HandshakeTimeout bounds the whole agent handshake: control stream
	// accept plus hello (default 5s).
	HandshakeTimeout time.Duration
	// MaxHandshakes caps unauthenticated agent connections: from the
	// upgrade until the agent is registered or refused.
	MaxHandshakes int
	// MaxHandshakesPerIP caps those per client address (IPv6: per /64).
	MaxHandshakesPerIP int
	// MaxAgents caps registered agents.
	MaxAgents int
	// MaxAgentsPerIP caps agent connections per client address (IPv6: per
	// /64), handshakes included.
	MaxAgentsPerIP int
	// MaxAgentsPerKey caps registered agents per API key.
	MaxAgentsPerKey int
	// MaxAgentsPerOwner caps registered agents per account.
	MaxAgentsPerOwner int
	// AuthorizePerMinute is the per-address rate of agent connection
	// attempts (each one authorizes with the site once), with an equal burst.
	AuthorizePerMinute int
	// ClientIPHeader names a header carrying the client address (for
	// example CF-Connecting-IP) set by the proxy chain. It is read only when
	// the TCP peer is a private or loopback address; empty uses the
	// rightmost X-Forwarded-For entry.
	ClientIPHeader string
	// MaxResponseBytes caps a non-SSE response body from an agent.
	MaxResponseBytes int64
	// MaxStreamResponseBytes caps an SSE response body from an agent.
	MaxStreamResponseBytes int64
	// StreamCloseTimeout bounds how long a finished request stream waits
	// for the agent's close (and may hold its buffered data) before it is
	// torn down (default 5s).
	StreamCloseTimeout time.Duration
	Logger             *slog.Logger
}

const (
	maxListingsPerAgent = 50
	maxAgentConcurrency = 256
	writeTimeout        = 30 * time.Second
	// Agent response header limits (status line + header block).
	maxResponseHeaderBytes  = 64 << 10
	maxResponseHeaderFields = 100
	responseBufferSize      = 32 << 10
	// maxReauthorizeFailures consecutive failed re-authorizations (site
	// down, malformed answers) disconnect an agent: revocation fails closed.
	maxReauthorizeFailures = 3
	sseContentType         = "text/event-stream; charset=utf-8"
	// maxTrackedAddresses caps the per-address rate-limit buckets; when it
	// is full the least recently used bucket is evicted.
	maxTrackedAddresses = 100_000
	// bucketSweepBatch idle buckets at most are dropped per admission.
	bucketSweepBatch = 64
	// revokedTTL keeps revoked keys and owners to refuse agents whose
	// authorization raced the revocation.
	revokedTTL = 10 * time.Minute
)

var (
	listingIDPattern  = regexp.MustCompile(`^[A-Za-z0-9_-]{1,64}$`)
	apiKeyPattern     = regexp.MustCompile(`^bhk_[0-9a-f]{40}$`)
	keyHashPattern    = regexp.MustCompile(`^[0-9a-f]{64}$`)
	headerNamePattern = regexp.MustCompile(`^[A-Za-z0-9-]{1,64}$`)
)

// ValidateSecret rejects missing, short and placeholder secrets.
func ValidateSecret(name, value string) error {
	switch {
	case value == "":
		return fmt.Errorf("%s is required", name)
	case strings.Contains(strings.ToLower(value), "replace-with"), value == "dev", value == "changeme", value == "secret":
		return fmt.Errorf("%s is a placeholder; generate one with openssl rand -hex 32", name)
	case len(value) < MinSecretLength:
		return fmt.Errorf("%s must be at least %d characters (openssl rand -hex 32)", name, MinSecretLength)
	}
	return nil
}

// Server holds the agent registry and serves the public and internal endpoints.
type Server struct {
	cfg      Config
	log      *slog.Logger
	mu       sync.Mutex
	agents   map[*agent]struct{}
	draining atomic.Bool
	forwards sync.WaitGroup
	// Revocation times by key hash and owner id, kept for revokedTTL.
	revokedKeys   map[string]time.Time
	revokedOwners map[string]time.Time

	// Admission: handshakes in progress (total and per address),
	// connections per address and the per-address attempt buckets (least
	// recently used first).
	admitMu        sync.Mutex
	handshakes     int
	handshakesByIP map[string]int
	connsByIP      map[string]int
	buckets        map[string]*list.Element
	bucketLRU      *list.List
	maxBuckets     int
}

type agent struct {
	id             string
	ownerID        string
	apiKey         string // memory only, for re-authorization; never logged
	keyHash        string // hex sha256 of apiKey
	authStarted    time.Time
	listings       []string
	maxConcurrency int
	session        *yamux.Session
	control        *relayproto.Conn
	inFlight       atomic.Int64
	draining       atomic.Bool
	closeOnce      sync.Once
	closed         chan struct{}
}

// New validates cfg and returns a Server. Weak or missing secrets are refused.
func New(cfg Config) (*Server, error) {
	if cfg.SiteURL == "" {
		return nil, errors.New("LMX_SITE_URL is required")
	}
	if err := ValidateSecret("RELAY_INTERNAL_SECRET", cfg.InternalSecret); err != nil {
		return nil, err
	}
	if err := ValidateSecret("RELAY_AUTHORIZE_SECRET", cfg.AuthorizeSecret); err != nil {
		return nil, err
	}
	if cfg.HTTPClient == nil {
		cfg.HTTPClient = &http.Client{Timeout: 10 * time.Second}
	}
	if cfg.Heartbeat <= 0 {
		cfg.Heartbeat = relayproto.DefaultHeartbeat
	}
	if cfg.Reauthorize <= 0 {
		cfg.Reauthorize = 5 * time.Minute
	}
	if cfg.ResponseHeaderTimeout <= 0 {
		cfg.ResponseHeaderTimeout = 120 * time.Second
	}
	if cfg.HandshakeTimeout <= 0 {
		cfg.HandshakeTimeout = DefaultHandshakeTimeout
	}
	if cfg.MaxHandshakes <= 0 {
		cfg.MaxHandshakes = DefaultMaxHandshakes
	}
	if cfg.MaxHandshakesPerIP <= 0 {
		cfg.MaxHandshakesPerIP = DefaultMaxHandshakesPerIP
	}
	if cfg.MaxAgents <= 0 {
		cfg.MaxAgents = DefaultMaxAgents
	}
	if cfg.MaxAgentsPerIP <= 0 {
		cfg.MaxAgentsPerIP = DefaultMaxAgentsPerIP
	}
	if cfg.MaxAgentsPerKey <= 0 {
		cfg.MaxAgentsPerKey = DefaultMaxAgentsPerKey
	}
	if cfg.MaxAgentsPerOwner <= 0 {
		cfg.MaxAgentsPerOwner = DefaultMaxAgentsPerOwner
	}
	if cfg.AuthorizePerMinute <= 0 {
		cfg.AuthorizePerMinute = DefaultAuthorizePerMinute
	}
	if cfg.MaxResponseBytes <= 0 {
		cfg.MaxResponseBytes = DefaultMaxResponseBytes
	}
	if cfg.MaxStreamResponseBytes <= 0 {
		cfg.MaxStreamResponseBytes = DefaultMaxStreamResponseBytes
	}
	if cfg.StreamCloseTimeout <= 0 {
		cfg.StreamCloseTimeout = DefaultStreamCloseTimeout
	}
	if cfg.ClientIPHeader != "" {
		if !headerNamePattern.MatchString(cfg.ClientIPHeader) {
			return nil, fmt.Errorf("LMX_RELAY_CLIENT_IP_HEADER %q is not a header name", cfg.ClientIPHeader)
		}
		cfg.ClientIPHeader = http.CanonicalHeaderKey(cfg.ClientIPHeader)
	}
	if cfg.Logger == nil {
		cfg.Logger = slog.New(slog.NewJSONHandler(io.Discard, nil))
	}
	cfg.SiteURL = strings.TrimRight(cfg.SiteURL, "/")
	return &Server{
		cfg: cfg, log: cfg.Logger, agents: map[*agent]struct{}{},
		revokedKeys: map[string]time.Time{}, revokedOwners: map[string]time.Time{},
		handshakesByIP: map[string]int{}, connsByIP: map[string]int{},
		buckets: map[string]*list.Element{}, bucketLRU: list.New(), maxBuckets: maxTrackedAddresses,
	}, nil
}

// PublicHandler serves the internet-facing listener: only the agent
// WebSocket endpoint and /healthz.
func (s *Server) PublicHandler() http.Handler {
	mux := http.NewServeMux()
	mux.HandleFunc(relayproto.AgentPath, s.handleAgent)
	mux.HandleFunc("/healthz", func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "application/json")
		_, _ = w.Write([]byte(`{"ok":true}`))
	})
	return mux
}

// InternalHandler serves the private listener the site calls: only /internal/*.
func (s *Server) InternalHandler() http.Handler {
	mux := http.NewServeMux()
	mux.HandleFunc("/internal/forward", s.internal(s.handleForward))
	mux.HandleFunc("/internal/agents", s.internal(s.handleAgents))
	mux.HandleFunc("/internal/agents/revoke", s.internal(s.handleRevoke))
	return mux
}

// Drain tells every agent to stop taking requests, refuses new agents and
// forwards, and waits (until ctx ends) for in-flight forwards to finish.
// Close the agents afterwards with Close.
func (s *Server) Drain(ctx context.Context) error {
	s.draining.Store(true)
	for _, ag := range s.snapshot() {
		ag.draining.Store(true)
		_ = ag.control.Write(relayproto.Notice{Type: relayproto.TypeDrain, Reason: "relay shutting down"}, writeTimeout)
	}
	done := make(chan struct{})
	go func() { s.forwards.Wait(); close(done) }()
	select {
	case <-done:
		return nil
	case <-ctx.Done():
		return ctx.Err()
	}
}

// Close disconnects every agent.
func (s *Server) Close() {
	for _, ag := range s.snapshot() {
		ag.close()
	}
}

func (s *Server) snapshot() []*agent {
	s.mu.Lock()
	defer s.mu.Unlock()
	out := make([]*agent, 0, len(s.agents))
	for ag := range s.agents {
		out = append(out, ag)
	}
	return out
}

// register adds ag unless the relay, its API key or its owner is at the
// agent limit, or its key or owner was revoked after its authorization
// started (the site may have approved it just before the revocation). It
// returns the refusal code and message.
func (s *Server) register(ag *agent) (string, string) {
	s.mu.Lock()
	defer s.mu.Unlock()
	if at, ok := s.revokedKeys[ag.keyHash]; ok && !at.Before(ag.authStarted) {
		return relayproto.CodeRevoked, "This API key was revoked while the agent was connecting."
	}
	if at, ok := s.revokedOwners[ag.ownerID]; ok && !at.Before(ag.authStarted) {
		return relayproto.CodeRevoked, "Hosting was revoked for this account while the agent was connecting."
	}
	if len(s.agents) >= s.cfg.MaxAgents {
		return relayproto.CodeTooManyAgents, "the relay is at its agent limit; retrying"
	}
	sameKey, sameOwner := 0, 0
	for other := range s.agents {
		if other.keyHash == ag.keyHash {
			sameKey++
		}
		if other.ownerID == ag.ownerID {
			sameOwner++
		}
	}
	if sameKey >= s.cfg.MaxAgentsPerKey {
		return relayproto.CodeTooManyAgents, fmt.Sprintf("this API key already has %d connected agents", s.cfg.MaxAgentsPerKey)
	}
	if sameOwner >= s.cfg.MaxAgentsPerOwner {
		return relayproto.CodeTooManyAgents, fmt.Sprintf("this account already has %d connected agents", s.cfg.MaxAgentsPerOwner)
	}
	s.agents[ag] = struct{}{}
	return "", ""
}

// registeredFull reports whether MaxAgents agents are registered.
func (s *Server) registeredFull() bool {
	s.mu.Lock()
	defer s.mu.Unlock()
	return len(s.agents) >= s.cfg.MaxAgents
}

func (s *Server) unregister(ag *agent) {
	s.mu.Lock()
	delete(s.agents, ag)
	s.mu.Unlock()
}

func (a *agent) close() {
	a.closeOnce.Do(func() {
		close(a.closed)
		_ = a.session.Close()
	})
}

func (a *agent) serves(listing string) bool {
	for _, id := range a.listings {
		if id == listing {
			return true
		}
	}
	return false
}

// ── Errors ──────────────────────────────────────────────────────────────────

// writeError writes a relay-originated error. Only the relay sets
// X-Lmx-Relay-Error; agent headers are never copied, so the site can tell
// relay failures from host responses.
func writeError(w http.ResponseWriter, status int, code, message string) {
	w.Header().Set(relayproto.HeaderRelayError, code)
	relayproto.WriteError(w, status, code, message)
}

// ── Internal authentication ─────────────────────────────────────────────────

func (s *Server) internal(next http.HandlerFunc) http.HandlerFunc {
	return func(w http.ResponseWriter, r *http.Request) {
		token, ok := strings.CutPrefix(r.Header.Get("Authorization"), "Bearer ")
		if !ok || !secretEqual(token, s.cfg.InternalSecret) {
			writeError(w, http.StatusUnauthorized, relayproto.CodeUnauthorized, "invalid relay secret")
			return
		}
		next(w, r)
	}
}

// secretEqual compares in constant time, including the length.
func secretEqual(got, want string) bool {
	a, b := sha256.Sum256([]byte(got)), sha256.Sum256([]byte(want))
	return subtle.ConstantTimeCompare(a[:], b[:]) == 1
}

// ── Admission ───────────────────────────────────────────────────────────────

// bucket is a per-address token bucket for connection attempts.
type bucket struct {
	key    string
	tokens float64
	last   time.Time
}

// admission holds a handshake slot (until endHandshake) and a connection
// slot for one address (until release).
type admission struct {
	s         *Server
	key       string
	handshake bool
}

// admit reserves a handshake slot and a connection slot for the address
// key and takes one authorize token, or reports why the connection is
// refused.
func (s *Server) admit(key string) (*admission, string, string) {
	s.admitMu.Lock()
	defer s.admitMu.Unlock()
	if s.handshakes >= s.cfg.MaxHandshakes {
		return nil, relayproto.CodeTooManyAgents, "the relay is handling too many agent handshakes; retrying"
	}
	if s.connsByIP[key] >= s.cfg.MaxAgentsPerIP {
		return nil, relayproto.CodeTooManyAgents, fmt.Sprintf("too many agent connections from this address (limit %d)", s.cfg.MaxAgentsPerIP)
	}
	if s.handshakesByIP[key] >= s.cfg.MaxHandshakesPerIP {
		return nil, relayproto.CodeTooManyAgents, fmt.Sprintf("too many agent handshakes in progress from this address (limit %d)", s.cfg.MaxHandshakesPerIP)
	}
	if code, message := s.takeToken(key, time.Now()); code != "" {
		return nil, code, message
	}
	s.handshakes++
	s.handshakesByIP[key]++
	s.connsByIP[key]++
	return &admission{s: s, key: key, handshake: true}, "", ""
}

// takeToken takes one attempt token from key's bucket. admitMu is held.
func (s *Server) takeToken(key string, now time.Time) (string, string) {
	capacity := float64(s.cfg.AuthorizePerMinute)
	// The list is ordered by last use. A bucket idle for a minute has
	// refilled, so dropping it changes nothing; a bounded batch per call
	// keeps admissions O(1).
	for range bucketSweepBatch {
		front := s.bucketLRU.Front()
		if front == nil || now.Sub(front.Value.(*bucket).last) < time.Minute {
			break
		}
		delete(s.buckets, front.Value.(*bucket).key)
		s.bucketLRU.Remove(front)
	}
	element := s.buckets[key]
	if element == nil {
		if len(s.buckets) >= s.maxBuckets {
			// Full: evict the least recently used address. At worst that
			// address gets a fresh burst; the connection and handshake caps
			// still bound it.
			front := s.bucketLRU.Front()
			delete(s.buckets, front.Value.(*bucket).key)
			s.bucketLRU.Remove(front)
		}
		element = s.bucketLRU.PushBack(&bucket{key: key, tokens: capacity, last: now})
		s.buckets[key] = element
	}
	b := element.Value.(*bucket)
	b.tokens = min(capacity, b.tokens+now.Sub(b.last).Minutes()*capacity)
	b.last = now
	s.bucketLRU.MoveToBack(element)
	if b.tokens < 1 {
		return relayproto.CodeRateLimited, "too many agent connection attempts from this address; slow down"
	}
	b.tokens--
	return "", ""
}

// endHandshake frees the handshake slot (registration done or failed).
func (a *admission) endHandshake() {
	a.s.admitMu.Lock()
	defer a.s.admitMu.Unlock()
	a.endHandshakeLocked()
}

func (a *admission) endHandshakeLocked() {
	if a.handshake {
		a.handshake = false
		a.s.handshakes--
		if a.s.handshakesByIP[a.key]--; a.s.handshakesByIP[a.key] <= 0 {
			delete(a.s.handshakesByIP, a.key)
		}
	}
}

// release frees both slots when the connection ends.
func (a *admission) release() {
	a.s.admitMu.Lock()
	defer a.s.admitMu.Unlock()
	a.endHandshakeLocked()
	if a.s.connsByIP[a.key]--; a.s.connsByIP[a.key] <= 0 {
		delete(a.s.connsByIP, a.key)
	}
}

// clientIP is the TCP peer, or, when the peer is a private or loopback
// address (the reverse proxy): the ipHeader value when that header is
// configured and holds an address, else the rightmost X-Forwarded-For
// entry, the one that proxy appended. Earlier entries are client-supplied.
func clientIP(r *http.Request, ipHeader string) string {
	host, _, err := net.SplitHostPort(r.RemoteAddr)
	if err != nil {
		host = r.RemoteAddr
	}
	peer, err := netip.ParseAddr(host)
	if err != nil {
		return host
	}
	peer = peer.Unmap()
	if !peer.IsPrivate() && !peer.IsLoopback() {
		return peer.String()
	}
	if ipHeader != "" {
		if addr, err := netip.ParseAddr(strings.TrimSpace(r.Header.Get(ipHeader))); err == nil {
			return addr.Unmap().String()
		}
	}
	values := r.Header.Values("X-Forwarded-For")
	if len(values) == 0 {
		return peer.String()
	}
	last := values[len(values)-1]
	entry := strings.TrimSpace(last[strings.LastIndex(last, ",")+1:])
	if forwarded, err := netip.ParseAddr(entry); err == nil {
		return forwarded.Unmap().String()
	}
	return peer.String()
}

// addressKey is the per-address limit key: the IPv4 address, or the IPv6
// /64 (one host usually controls a whole /64).
func addressKey(ip string) string {
	addr, err := netip.ParseAddr(ip)
	if err != nil || !addr.Is6() {
		return ip
	}
	prefix, err := addr.WithZone("").Prefix(64)
	if err != nil {
		return ip
	}
	return prefix.String()
}

// upgradeError checks the WebSocket handshake headers (as websocket.Accept
// does) so that requests that cannot upgrade never reach admission.
func upgradeError(r *http.Request) string {
	switch {
	case r.Method != http.MethodGet || !r.ProtoAtLeast(1, 1):
		return "a GET HTTP/1.1 WebSocket upgrade is required"
	case !headerHasToken(r.Header, "Connection", "upgrade", true), !headerHasToken(r.Header, "Upgrade", "websocket", true):
		return "a WebSocket upgrade is required"
	case r.Header.Get("Sec-WebSocket-Version") != "13":
		return "Sec-WebSocket-Version 13 is required"
	case !headerHasToken(r.Header, "Sec-WebSocket-Protocol", relayproto.Subprotocol, false):
		return "subprotocol " + relayproto.Subprotocol + " is required"
	}
	keys := r.Header.Values("Sec-WebSocket-Key")
	if len(keys) != 1 {
		return "one Sec-WebSocket-Key is required"
	}
	if key, err := base64.StdEncoding.DecodeString(strings.TrimSpace(keys[0])); err != nil || len(key) != 16 {
		return "invalid Sec-WebSocket-Key"
	}
	return ""
}

// headerHasToken reports whether a comma-separated header lists token.
func headerHasToken(header http.Header, name, token string, fold bool) bool {
	for _, value := range header.Values(name) {
		for _, item := range strings.Split(value, ",") {
			item = strings.TrimSpace(item)
			if item == token || (fold && strings.EqualFold(item, token)) {
				return true
			}
		}
	}
	return false
}

// ── Site authorize ──────────────────────────────────────────────────────────

type authorization struct {
	OwnerID  string `json:"ownerId"`
	Plan     string `json:"plan"`
	Listings []struct {
		ID             string `json:"id"`
		Status         string `json:"status"`
		MaxConcurrency int    `json:"maxConcurrency"`
	} `json:"listings"`
}

// authError is a definitive refusal from the site (401/403/404).
type authError struct{ code, message string }

func (e *authError) Error() string { return e.code + ": " + e.message }

func (s *Server) authorize(ctx context.Context, apiKey string, listings []string) (*authorization, error) {
	body, _ := json.Marshal(map[string]any{"apiKey": apiKey, "listingIds": listings})
	req, err := http.NewRequestWithContext(ctx, http.MethodPost, s.cfg.SiteURL+"/api/internal/relay/authorize", bytes.NewReader(body))
	if err != nil {
		return nil, err
	}
	req.Header.Set("Content-Type", "application/json")
	req.Header.Set("Authorization", "Bearer "+s.cfg.AuthorizeSecret)
	res, err := s.cfg.HTTPClient.Do(req)
	if err != nil {
		return nil, fmt.Errorf("authorize request failed: %w", err)
	}
	defer res.Body.Close()
	data, _ := io.ReadAll(io.LimitReader(res.Body, 1<<20))
	switch res.StatusCode {
	case http.StatusOK:
		var auth authorization
		if err := json.Unmarshal(data, &auth); err != nil || auth.OwnerID == "" {
			return nil, errors.New("authorize returned a malformed body")
		}
		return &auth, nil
	case http.StatusUnauthorized, http.StatusForbidden, http.StatusNotFound:
		var refusal struct {
			Code    string `json:"code"`
			Message string `json:"message"`
			Error   string `json:"error"`
		}
		_ = json.Unmarshal(data, &refusal)
		code := refusal.Code
		if code == "" {
			code = map[int]string{401: relayproto.CodeUnauthorized, 403: relayproto.CodeProRequired, 404: relayproto.CodeListingNotFound}[res.StatusCode]
		}
		message := refusal.Message
		if message == "" {
			message = refusal.Error
		}
		if message == "" {
			message = refusalMessage(code)
		}
		return nil, &authError{code: code, message: message}
	default:
		return nil, fmt.Errorf("authorize returned HTTP %d", res.StatusCode)
	}
}

func refusalMessage(code string) string {
	switch code {
	case relayproto.CodeProRequired:
		return "Hosting rentals requires a LocalMaxxing Pro plan."
	case relayproto.CodeUnauthorized:
		return "The API key is invalid or revoked."
	case relayproto.CodeListingNotOwned:
		return "A listing belongs to another account."
	case relayproto.CodeListingNotFound:
		return "A listing does not exist."
	}
	return "Authorization was refused."
}

// ── Agent connections ───────────────────────────────────────────────────────

// yamuxConfig is the relay's side of the session. The agent opens exactly
// one stream (control), so the accept backlog is 1: further streams opened
// before the handshake finishes are reset by yamux, and after it
// refuseAgentStreams disconnects the agent. yamux's stream window cannot go
// below its 256 KiB initial window, which already is the default maximum.
// A closed stream waits StreamCloseTimeout (not yamux's 5 minutes) for the
// agent's close before it is freed with whatever it buffered.
func (s *Server) yamuxConfig() *yamux.Config {
	config := relayproto.YamuxConfig()
	config.AcceptBacklog = 1
	config.StreamCloseTimeout = s.cfg.StreamCloseTimeout
	return config
}

func (s *Server) handleAgent(w http.ResponseWriter, r *http.Request) {
	apiKey, ok := strings.CutPrefix(r.Header.Get("Authorization"), "Bearer ")
	if !ok || !apiKeyPattern.MatchString(apiKey) {
		writeError(w, http.StatusUnauthorized, relayproto.CodeUnauthorized, "Authorization: Bearer bhk_<40 hex> is required")
		return
	}
	if reason := upgradeError(r); reason != "" {
		writeError(w, http.StatusUpgradeRequired, "upgrade_required", reason)
		return
	}
	if s.draining.Load() {
		writeError(w, http.StatusServiceUnavailable, relayproto.CodeRelayDraining, "relay is shutting down")
		return
	}
	ip := clientIP(r, s.cfg.ClientIPHeader)
	if s.registeredFull() {
		s.log.Info("agent refused", "remote", ip, "code", relayproto.CodeTooManyAgents)
		writeError(w, http.StatusTooManyRequests, relayproto.CodeTooManyAgents, "the relay is at its agent limit; retrying")
		return
	}
	admitted, code, message := s.admit(addressKey(ip))
	if admitted == nil {
		s.log.Info("agent refused", "remote", ip, "code", code)
		writeError(w, http.StatusTooManyRequests, code, message)
		return
	}
	defer admitted.release()
	ws, err := websocket.Accept(w, r, &websocket.AcceptOptions{Subprotocols: []string{relayproto.Subprotocol}})
	if err != nil {
		return
	}
	ctx, cancel := context.WithCancel(r.Context())
	defer cancel()
	session, err := yamux.Client(relayproto.NetConn(ctx, ws), s.yamuxConfig())
	if err != nil {
		_ = ws.Close(websocket.StatusInternalError, "session setup failed")
		return
	}
	defer session.Close()
	ag, err := s.handshake(ctx, session, apiKey)
	admitted.endHandshake()
	if err != nil {
		s.log.Info("agent rejected", "remote", ip, "reason", err.Error())
		return
	}
	defer s.unregister(ag)
	s.log.Info("agent connected", "agent", ag.id, "owner", ag.ownerID, "listings", ag.listings, "maxConcurrency", ag.maxConcurrency, "remote", ip)
	go s.refuseAgentStreams(ag)
	go s.reauthorizeLoop(ag)
	reason := s.controlLoop(ag)
	ag.close()
	s.log.Info("agent disconnected", "agent", ag.id, "owner", ag.ownerID, "reason", reason)
}

// refuseAgentStreams disconnects an agent that opens a stream after the
// control stream: the protocol has none, and an unread stream would pin a
// receive window of relay memory.
func (s *Server) refuseAgentStreams(ag *agent) {
	stream, err := ag.session.AcceptStream()
	if err != nil {
		return
	}
	_ = stream.Close()
	s.log.Warn("agent opened an unexpected stream; disconnecting", "agent", ag.id, "owner", ag.ownerID)
	ag.close()
}

func (s *Server) handshake(ctx context.Context, session *yamux.Session, apiKey string) (*agent, error) {
	// One deadline for the control stream and the hello together.
	deadline := time.Now().Add(s.cfg.HandshakeTimeout)
	acceptCtx, cancel := context.WithDeadline(ctx, deadline)
	defer cancel()
	stream, err := session.AcceptStreamWithContext(acceptCtx)
	if err != nil {
		return nil, fmt.Errorf("no control stream: %w", err)
	}
	control := relayproto.NewConn(stream)
	reject := func(code, message string) error {
		_ = control.Write(relayproto.Notice{Type: relayproto.TypeError, Code: code, Message: message}, writeTimeout)
		_ = control.Close()
		return errors.New(code)
	}
	kind, raw, err := control.Read(max(time.Until(deadline), time.Millisecond))
	if err != nil {
		return nil, fmt.Errorf("no hello: %w", err)
	}
	var hello relayproto.Hello
	if kind != relayproto.TypeHello || json.Unmarshal(raw, &hello) != nil {
		return nil, reject(relayproto.CodeBadHello, "the first control message must be hello")
	}
	if hello.Proto != relayproto.Proto {
		return nil, reject(relayproto.CodeProtoUnsupported, fmt.Sprintf("relay speaks proto %d; update lmx (lmx update)", relayproto.Proto))
	}
	if len(hello.Listings) == 0 || len(hello.Listings) > maxListingsPerAgent {
		return nil, reject(relayproto.CodeBadHello, fmt.Sprintf("hello must name 1-%d listings", maxListingsPerAgent))
	}
	for _, id := range hello.Listings {
		if !listingIDPattern.MatchString(id) {
			return nil, reject(relayproto.CodeListingNotFound, "invalid listing id")
		}
	}
	authStarted := time.Now()
	auth, err := s.authorize(ctx, apiKey, hello.Listings)
	if err != nil {
		var refused *authError
		if errors.As(err, &refused) {
			return nil, reject(refused.code, refused.message)
		}
		s.log.Warn("authorize unavailable", "error", err.Error())
		return nil, reject(relayproto.CodeSiteUnavailable, "LocalMaxxing could not authorize the agent right now; retrying")
	}
	keyHash := sha256.Sum256([]byte(apiKey))
	ag := &agent{
		id:             newAgentID(),
		ownerID:        auth.OwnerID,
		apiKey:         apiKey,
		keyHash:        hex.EncodeToString(keyHash[:]),
		authStarted:    authStarted,
		maxConcurrency: min(max(hello.MaxConcurrency, 1), maxAgentConcurrency),
		session:        session,
		control:        control,
		closed:         make(chan struct{}),
	}
	statuses := make([]relayproto.ListingStatus, 0, len(auth.Listings))
	for _, listing := range auth.Listings {
		ag.listings = append(ag.listings, listing.ID)
		statuses = append(statuses, relayproto.ListingStatus{ID: listing.ID, Status: listing.Status})
	}
	if code, message := s.register(ag); code != "" {
		return nil, reject(code, message)
	}
	welcome := relayproto.Welcome{Type: relayproto.TypeWelcome, AgentID: ag.id, HeartbeatSec: int(s.cfg.Heartbeat / time.Second), Listings: statuses}
	if welcome.HeartbeatSec < 1 {
		welcome.HeartbeatSec = 1
	}
	if err := control.Write(welcome, writeTimeout); err != nil {
		s.unregister(ag)
		return nil, fmt.Errorf("welcome failed: %w", err)
	}
	return ag, nil
}

// controlLoop answers pings, sends heartbeats and closes after missed ones.
// It returns when the agent is gone.
func (s *Server) controlLoop(ag *agent) string {
	var lastSeen atomic.Int64
	lastSeen.Store(time.Now().UnixNano())
	readErr := make(chan error, 1)
	go func() {
		for {
			kind, _, err := ag.control.Read(0)
			if err != nil {
				readErr <- err
				return
			}
			lastSeen.Store(time.Now().UnixNano())
			switch kind {
			case relayproto.TypePing:
				_ = ag.control.Write(relayproto.Notice{Type: relayproto.TypePong}, writeTimeout)
			case relayproto.TypeDrain:
				// Agent is shutting down: route nothing new to it.
				ag.draining.Store(true)
				s.log.Info("agent draining", "agent", ag.id)
			}
		}
	}()
	ticker := time.NewTicker(s.cfg.Heartbeat)
	defer ticker.Stop()
	for {
		select {
		case <-ag.closed:
			return "closed"
		case <-ag.session.CloseChan():
			return "session closed"
		case err := <-readErr:
			if errors.Is(err, io.EOF) {
				return "control stream closed"
			}
			return "control read failed"
		case <-ticker.C:
			if time.Since(time.Unix(0, lastSeen.Load())) > relayproto.MissedHeartbeats*s.cfg.Heartbeat {
				return "heartbeat timeout"
			}
			if err := ag.control.Write(relayproto.Notice{Type: relayproto.TypePing}, writeTimeout); err != nil {
				return "heartbeat write failed"
			}
		}
	}
}

// reauthorizeLoop re-checks the agent with the site. A refusal revokes it
// at once; maxReauthorizeFailures consecutive failures to get an answer
// revoke it too (it reconnects and authorizes again once the site is back).
func (s *Server) reauthorizeLoop(ag *agent) {
	ticker := time.NewTicker(s.cfg.Reauthorize)
	defer ticker.Stop()
	failures := 0
	for {
		select {
		case <-ag.closed:
			return
		case <-ag.session.CloseChan():
			return
		case <-ticker.C:
			ctx, cancel := context.WithTimeout(context.Background(), 15*time.Second)
			_, err := s.authorize(ctx, ag.apiKey, ag.listings)
			cancel()
			var refused *authError
			if errors.As(err, &refused) {
				s.revoke(ag, refused.code, refused.message)
				return
			}
			if err == nil {
				failures = 0
				continue
			}
			failures++
			if failures >= maxReauthorizeFailures {
				s.log.Warn("reauthorize failed repeatedly; disconnecting agent", "agent", ag.id, "failures", failures, "error", err.Error())
				s.revoke(ag, relayproto.CodeSiteUnavailable, "LocalMaxxing could not re-authorize this agent; reconnecting")
				return
			}
			s.log.Warn("reauthorize unavailable", "agent", ag.id, "failures", failures, "error", err.Error())
		}
	}
}

func (s *Server) revoke(ag *agent, code, message string) {
	s.log.Info("agent revoked", "agent", ag.id, "owner", ag.ownerID, "code", code)
	ag.draining.Store(true)
	_ = ag.control.Write(relayproto.Notice{Type: relayproto.TypeRevoked, Code: code, Message: message}, 5*time.Second)
	_ = ag.control.Close()
	// Give the revoked notice a moment to reach the agent before tearing down.
	time.AfterFunc(time.Second, ag.close)
}

// pick returns the least-loaded live agent for a listing with a free slot
// (below its advertised maxConcurrency) and reserves the slot. busy reports
// that agents serve the listing but all are full.
func (s *Server) pick(listing string) (best *agent, busy bool) {
	s.mu.Lock()
	defer s.mu.Unlock()
	for ag := range s.agents {
		if ag.draining.Load() || ag.session.IsClosed() || !ag.serves(listing) {
			continue
		}
		load := ag.inFlight.Load()
		if load >= int64(ag.maxConcurrency) {
			busy = true
			continue
		}
		if best == nil || load < best.inFlight.Load() {
			best = ag
		}
	}
	if best != nil {
		best.inFlight.Add(1)
	}
	return best, busy
}

// ── Internal endpoints ──────────────────────────────────────────────────────

func (s *Server) handleForward(w http.ResponseWriter, r *http.Request) {
	if r.Method != http.MethodPost {
		writeError(w, http.StatusMethodNotAllowed, "method_not_allowed", "POST only")
		return
	}
	s.forwards.Add(1)
	defer s.forwards.Done()
	started := time.Now()
	listing, sessionID := r.Header.Get(relayproto.HeaderListing), r.Header.Get(relayproto.HeaderSession)
	method, path := strings.ToUpper(r.Header.Get(relayproto.HeaderMethod)), r.Header.Get(relayproto.HeaderPath)
	if !listingIDPattern.MatchString(listing) {
		writeError(w, http.StatusBadRequest, "bad_request", "X-Lmx-Listing is required")
		return
	}
	if !relayproto.AllowedRoute(method, path) {
		writeError(w, http.StatusNotFound, "not_found", "route is not available on rentals")
		return
	}
	if s.draining.Load() {
		writeError(w, http.StatusServiceUnavailable, "no_agent", "relay is shutting down")
		return
	}
	body, err := io.ReadAll(http.MaxBytesReader(w, r.Body, relayproto.MaxRequestBody))
	if err != nil {
		writeError(w, http.StatusRequestEntityTooLarge, "body_too_large", "request body is too large")
		return
	}
	ag, busy := s.pick(listing)
	if ag == nil {
		message := "no host is connected for this listing"
		if busy {
			message = "every host for this listing is at its concurrency limit"
		}
		writeError(w, http.StatusServiceUnavailable, "no_agent", message)
		return
	}
	defer ag.inFlight.Add(-1)
	status, bytesOut, outcome, abort := s.forward(w, r, ag, method, path, listing, sessionID, body)
	s.log.Info("forward", "listing", listing, "session", sessionID, "agent", ag.id, "path", path,
		"status", status, "outcome", outcome, "bytesOut", bytesOut, "durationMs", time.Since(started).Milliseconds())
	if abort {
		// The body did not reach its end: abort the response so the site
		// sees a failed transfer instead of a truncated body that looks
		// complete.
		panic(http.ErrAbortHandler)
	}
}

const outcomeTooLarge = "response_too_large"

// headerLimitReader fails reads past limit bytes while limit >= 0.
type headerLimitReader struct {
	r     io.Reader
	n     int64
	limit int64
}

var errResponseHeadersTooLarge = errors.New("agent response headers too large")

func (l *headerLimitReader) Read(p []byte) (int, error) {
	if l.limit >= 0 {
		if l.n >= l.limit {
			return 0, errResponseHeadersTooLarge
		}
		if remaining := l.limit - l.n; int64(len(p)) > remaining {
			p = p[:remaining]
		}
	}
	n, err := l.r.Read(p)
	l.n += int64(n)
	return n, err
}

// responseContentType maps an agent's Content-Type onto the two types the
// relay emits; anything but an event stream is sent as JSON.
func responseContentType(agentType string) (string, bool) {
	mediaType, _, _ := strings.Cut(agentType, ";")
	if strings.EqualFold(strings.TrimSpace(mediaType), "text/event-stream") {
		return sseContentType, true
	}
	return "application/json", false
}

// forward relays one request to ag. abort reports that the response must be
// aborted: its headers are out but the body did not reach its end (or the
// renter is gone).
func (s *Server) forward(w http.ResponseWriter, r *http.Request, ag *agent, method, path, listing, sessionID string, body []byte) (status int, written int64, outcome string, abort bool) {
	stream, err := ag.session.OpenStream()
	if err != nil {
		writeError(w, http.StatusBadGateway, "agent_failed", "could not open a stream to the host")
		return http.StatusBadGateway, 0, "open_failed", false
	}
	// Every path but a complete response closes the stream at once, without
	// draining it (the agent sees the close and cancels); yamux frees it
	// after StreamCloseTimeout if the agent never closes its side. A
	// complete response hands the stream to finishStream.
	finished := false
	defer func() {
		if !finished {
			_ = stream.Close()
		}
	}()
	// Renter gone → close the stream (the agent cancels) and stop reading.
	stop := context.AfterFunc(r.Context(), func() {
		_ = stream.Close()
		_ = stream.SetReadDeadline(time.Now())
	})
	defer stop()

	var reqBody io.Reader
	if len(body) > 0 {
		reqBody = bytes.NewReader(body)
	}
	req, err := http.NewRequest(method, "http://lmx-agent"+path, reqBody)
	if err != nil {
		writeError(w, http.StatusBadRequest, "bad_request", "invalid request")
		return http.StatusBadRequest, 0, "bad_request", false
	}
	for _, name := range []string{"Content-Type", "Accept"} {
		if value := r.Header.Get(name); value != "" {
			req.Header.Set(name, value)
		}
	}
	req.Header.Set(relayproto.HeaderListing, listing)
	req.Header.Set(relayproto.HeaderSession, sessionID)
	_ = stream.SetWriteDeadline(time.Now().Add(writeTimeout))
	if err := req.Write(stream); err != nil {
		writeError(w, http.StatusBadGateway, "agent_failed", "could not send the request to the host")
		return http.StatusBadGateway, 0, "write_failed", false
	}
	_ = stream.SetWriteDeadline(time.Time{})

	// http.ReadResponse does not bound the status line or headers: cap the
	// bytes read while parsing them (the reader may prefetch one buffer of
	// body, so the exact header size is checked after parsing).
	_ = stream.SetReadDeadline(time.Now().Add(s.cfg.ResponseHeaderTimeout))
	limited := &headerLimitReader{r: stream, limit: maxResponseHeaderBytes + responseBufferSize}
	buffered := bufio.NewReaderSize(limited, responseBufferSize)
	res, err := http.ReadResponse(buffered, req)
	if err != nil {
		if r.Context().Err() != nil {
			return 499, 0, "renter_cancelled", true
		}
		var netErr net.Error
		if (errors.As(err, &netErr) && netErr.Timeout()) || errors.Is(err, yamux.ErrTimeout) {
			writeError(w, http.StatusGatewayTimeout, "agent_timeout", "the host did not respond in time")
			return http.StatusGatewayTimeout, 0, "timeout", false
		}
		writeError(w, http.StatusBadGateway, "agent_failed", "the host closed the request without a valid response")
		return http.StatusBadGateway, 0, "read_failed", false
	}
	// res.Body is never closed: Close would read the rest of an unfinished
	// body from the agent. The stream is closed instead.
	headerFields := 0
	for _, values := range res.Header {
		headerFields += len(values)
	}
	if limited.n-int64(buffered.Buffered()) > maxResponseHeaderBytes || headerFields > maxResponseHeaderFields {
		writeError(w, http.StatusBadGateway, "agent_failed", "the host sent oversized response headers")
		return http.StatusBadGateway, 0, "headers_too_large", false
	}
	if res.StatusCode < 200 || res.StatusCode > 599 {
		writeError(w, http.StatusBadGateway, "agent_failed", "the host sent an invalid status")
		return http.StatusBadGateway, 0, "bad_status", false
	}
	limited.limit = -1
	_ = stream.SetReadDeadline(time.Time{})

	// Only the normalized type and length are passed on; no other agent
	// header (in particular no X-Lmx-*) reaches the site.
	contentType, isStream := responseContentType(res.Header.Get("Content-Type"))
	maxBody := s.cfg.MaxResponseBytes
	if isStream {
		maxBody = s.cfg.MaxStreamResponseBytes
	}
	if res.ContentLength > maxBody {
		writeError(w, http.StatusBadGateway, outcomeTooLarge, "the host's response is too large")
		return http.StatusBadGateway, 0, "content_length_too_large", false
	}
	w.Header().Set("Content-Type", contentType)
	if res.ContentLength >= 0 {
		w.Header().Set("Content-Length", strconv.FormatInt(res.ContentLength, 10))
	}
	w.Header().Set("Cache-Control", "no-store")
	w.WriteHeader(res.StatusCode)
	flusher := http.NewResponseController(w)
	_ = flusher.Flush()
	buf := make([]byte, 32<<10)
	for {
		n, readErr := res.Body.Read(buf)
		if n > 0 {
			if written+int64(n) > maxBody {
				return res.StatusCode, written, outcomeTooLarge, true
			}
			if _, err := w.Write(buf[:n]); err != nil {
				return res.StatusCode, written, "renter_write_failed", true
			}
			written += int64(n)
			_ = flusher.Flush()
		}
		if readErr == io.EOF {
			finished = true
			go s.finishStream(ag, stream, buffered.Buffered())
			return res.StatusCode, written, "ok", false
		}
		if readErr != nil {
			if r.Context().Err() != nil {
				return res.StatusCode, written, "renter_cancelled", true
			}
			return res.StatusCode, written, "agent_stream_failed", true
		}
	}
}

// finishStream ends the stream of a complete response. The agent may only
// close it now: bytes after the response (already buffered, or arriving
// before its close) would sit in relay memory unread, so they disconnect
// the agent. An agent that neither closes nor sends within
// StreamCloseTimeout is closed on (and freed by yamux one timeout later).
func (s *Server) finishStream(ag *agent, stream *yamux.Stream, buffered int) {
	defer stream.Close()
	if buffered == 0 {
		_ = stream.SetReadDeadline(time.Now().Add(s.cfg.StreamCloseTimeout))
		var probe [1]byte
		if n, _ := stream.Read(probe[:]); n == 0 {
			return
		}
	}
	s.log.Warn("agent sent data after its response; disconnecting", "agent", ag.id, "owner", ag.ownerID)
	ag.close()
}

func (s *Server) handleAgents(w http.ResponseWriter, r *http.Request) {
	listing := r.URL.Query().Get("listing")
	if listing == "" {
		writeError(w, http.StatusBadRequest, "bad_request", "listing query parameter is required")
		return
	}
	agents, inFlight := 0, int64(0)
	for _, ag := range s.snapshot() {
		if !ag.serves(listing) {
			continue
		}
		inFlight += ag.inFlight.Load()
		if !ag.draining.Load() && !ag.session.IsClosed() {
			agents++
		}
	}
	writeJSON(w, map[string]any{"listing": listing, "agents": agents, "inFlight": inFlight})
}

// handleRevoke disconnects every agent of an owner (plan downgrade) or of
// one API key (key deleted), identified by the sha256 hex of the key. The
// revocation is kept for revokedTTL, so an agent whose authorization was
// already in flight is refused when it registers.
func (s *Server) handleRevoke(w http.ResponseWriter, r *http.Request) {
	if r.Method != http.MethodPost {
		writeError(w, http.StatusMethodNotAllowed, "method_not_allowed", "POST only")
		return
	}
	var body struct {
		OwnerID    string `json:"ownerId"`
		APIKeyHash string `json:"apiKeyHash"`
	}
	err := json.NewDecoder(io.LimitReader(r.Body, 64<<10)).Decode(&body)
	if err != nil || (body.OwnerID == "") == (body.APIKeyHash == "") || len(body.OwnerID) > 128 || (body.APIKeyHash != "" && !keyHashPattern.MatchString(body.APIKeyHash)) {
		writeError(w, http.StatusBadRequest, "bad_request", "send exactly one of ownerId or apiKeyHash (sha256 hex of the API key)")
		return
	}
	code := relayproto.CodeProRequired
	if body.APIKeyHash != "" {
		code = relayproto.CodeUnauthorized
	}
	now := time.Now()
	var targets []*agent
	s.mu.Lock()
	for _, revoked := range []map[string]time.Time{s.revokedKeys, s.revokedOwners} {
		for id, at := range revoked {
			if now.Sub(at) > revokedTTL {
				delete(revoked, id)
			}
		}
	}
	if body.APIKeyHash != "" {
		s.revokedKeys[body.APIKeyHash] = now
	} else {
		s.revokedOwners[body.OwnerID] = now
	}
	for ag := range s.agents {
		if (body.OwnerID != "" && ag.ownerID == body.OwnerID) || (body.APIKeyHash != "" && ag.keyHash == body.APIKeyHash) {
			targets = append(targets, ag)
		}
	}
	s.mu.Unlock()
	for _, ag := range targets {
		s.revoke(ag, code, refusalMessage(code))
	}
	writeJSON(w, map[string]any{"revoked": len(targets)})
}

func writeJSON(w http.ResponseWriter, value any) {
	w.Header().Set("Content-Type", "application/json")
	_ = json.NewEncoder(w).Encode(value)
}

func newAgentID() string {
	var b [8]byte
	_, _ = rand.Read(b[:])
	return "agt_" + hex.EncodeToString(b[:])
}
