package main

import (
	"bufio"
	"bytes"
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"math/rand/v2"
	"net"
	"net/http"
	"net/netip"
	"net/url"
	"os"
	"os/signal"
	"strconv"
	"strings"
	"sync/atomic"
	"syscall"
	"time"

	"github.com/LottoLottoLotto/localmaxxing-cli/internal/relayproto"
	"github.com/coder/websocket"
	"github.com/hashicorp/yamux"
)

// Rentals: `lmx rent host` serves listings from this machine through the
// LocalMaxxing relay over one outbound connection (no public endpoint), and
// `lmx rent connect` exposes a rented listing as a local OpenAI-compatible
// base URL. The site authorizes, meters and bills; the relay only carries
// bytes; the host enforces its own route allowlist and model list.

func handleRent(action string, args cliArgs) error {
	if action == "host" || action == "connect" {
		// Said on stderr, so --json output on stdout stays as it is.
		fmt.Fprintln(os.Stderr, "lmx rent is experimental: rentals are not finished and the relay they use may not be in service yet.")
	}
	switch action {
	case "host":
		return handleRentHost(args)
	case "connect":
		return handleRentConnect(args)
	case "list", "ls":
		return handleRentList(args)
	case "":
		return cliError{"missing_subcommand", "lmx rent needs a subcommand.", []string{"Use lmx rent host <listingId>, lmx rent connect <listingId>, or lmx rent list [--mine]."}, nil}
	}
	return cliError{"unknown_subcommand", "Unknown rent subcommand: " + action, []string{"Use lmx rent host, lmx rent connect, or lmx rent list."}, map[string]any{"subcommand": action}}
}

// optValues returns every value of a repeatable option, also splitting
// comma-separated values (--model a --model b, or --model a,b).
func optValues(args cliArgs, key string) []string {
	raw := args.repeated[key]
	if len(raw) == 0 && opt(args, key) != "" {
		raw = []string{opt(args, key)}
	}
	values := []string{}
	for _, value := range raw {
		for _, item := range strings.Split(value, ",") {
			if item = strings.TrimSpace(item); item != "" && !sliceContainsString(values, item) {
				values = append(values, item)
			}
		}
	}
	return values
}

// rentRelayURL is --relay-url, LMX_RELAY_URL, else the configured API origin
// with ws(s) and the relay agent path. ws:// would send the API key in
// cleartext, so it is refused unless the relay is on this machine.
func rentRelayURL(args cliArgs) (string, error) {
	relayURL := firstNonEmpty(opt(args, "relay-url"), os.Getenv("LMX_RELAY_URL"))
	if relayURL != "" {
		parsed, err := url.Parse(relayURL)
		if err != nil || (parsed.Scheme != "ws" && parsed.Scheme != "wss") || parsed.Host == "" {
			return "", cliError{"invalid_option", "--relay-url must be a ws:// or wss:// URL.", []string{"Example: --relay-url wss://www.localmaxxing.com" + relayproto.AgentPath}, nil}
		}
	} else {
		parsed, err := url.Parse(apiURL(args))
		if err != nil || parsed.Host == "" {
			return "", cliError{"invalid_option", "--api-url is not a valid URL.", nil, nil}
		}
		scheme := "wss"
		if parsed.Scheme == "http" {
			scheme = "ws"
		}
		relayURL = scheme + "://" + parsed.Host + relayproto.AgentPath
	}
	parsed, _ := url.Parse(relayURL)
	if parsed.Scheme == "ws" && !isLoopbackHost(parsed.Hostname()) {
		return "", cliError{"insecure_relay_url", "Refusing " + relayURL + ": ws:// would send your API key unencrypted. ws:// is allowed only for localhost.", []string{
			"Use wss://, e.g. --relay-url wss://" + parsed.Host + relayproto.AgentPath + ", or an https:// --api-url.",
		}, nil}
	}
	return relayURL, nil
}

// isLoopbackHost reports whether host (no port, no brackets) is localhost or
// a loopback IP.
func isLoopbackHost(host string) bool {
	if strings.EqualFold(host, "localhost") {
		return true
	}
	addr, err := netip.ParseAddr(host)
	return err == nil && addr.Unmap().IsLoopback()
}

// ── rent host ───────────────────────────────────────────────────────────────

type rentHostConfig struct {
	RelayURL       string
	APIKey         string
	Listings       []string
	Target         string // OpenAI-compatible base URL, no /v1
	TargetKey      string // bearer for forwarded renter requests (--target-key, LMX_RENT_TARGET_KEY)
	DiscoveryKey   string // bearer for model discovery only (EMX_API_KEY)
	Models         []string
	MaxConcurrency int
	SiteURL        string // for hints
	DrainTimeout   time.Duration
	BackoffBase    time.Duration
	BackoffMax     time.Duration
	Client         *http.Client
	Status         func(event string, fields map[string]any)
}

func handleRentHost(args cliArgs) error {
	listings := args.positional[min(2, len(args.positional)):]
	if len(listings) == 0 {
		return cliError{"missing_listing", "lmx rent host needs at least one listing ID.", []string{"Find your listing IDs with lmx rent list --mine.", "Then run lmx rent host <listingId>."}, nil}
	}
	key := apiKey(args)
	if key == "" {
		return missingAPIKey("lmx rent host authenticates with your LocalMaxxing API key.")
	}
	relayURL, err := rentRelayURL(args)
	if err != nil {
		return err
	}
	maxConcurrency, err := intOption(args, 4, 1, "max-concurrency")
	if err != nil {
		return err
	}
	target := openAIBaseURL(firstNonEmpty(opt(args, "target"), gatewayURL(args)))
	if parsed, err := url.Parse(target); err != nil || (parsed.Scheme != "http" && parsed.Scheme != "https") || parsed.Host == "" {
		return cliError{"invalid_option", "--target must be an http(s) base URL, e.g. http://127.0.0.1:8000.", nil, nil}
	}
	cfg := rentHostConfig{
		RelayURL: relayURL, APIKey: key, Listings: listings, Target: target,
		TargetKey: firstNonEmpty(opt(args, "target-key"), os.Getenv("LMX_RENT_TARGET_KEY")), DiscoveryKey: os.Getenv("EMX_API_KEY"),
		MaxConcurrency: maxConcurrency, SiteURL: apiURL(args),
		Client: &http.Client{}, Status: func(event string, fields map[string]any) { printStatus(args, event, fields) },
	}
	cfg.Models = optValues(args, "model")
	if len(cfg.Models) == 0 {
		models, err := rentTargetModels(cfg)
		if err != nil {
			return err
		}
		cfg.Models = models
	}
	if cfg.TargetKey == "" && cfg.DiscoveryKey != "" {
		cfg.Status("rent_host_target_key_unset", map[string]any{
			"note": "EMX_API_KEY is used only to discover models; renter requests reach --target without a key. If the target needs one, set LMX_RENT_TARGET_KEY (or --target-key) to a client key limited to the rented models: emx key add rentals --models '" + strings.Join(cfg.Models, ",") + "'.",
		})
	}

	ctx, stop := context.WithCancel(context.Background())
	defer stop()
	signals := make(chan os.Signal, 2)
	signal.Notify(signals, os.Interrupt, syscall.SIGTERM)
	defer signal.Stop(signals)
	go func() {
		<-signals
		cfg.Status("rent_host_stopping", map[string]any{"next": "Finishing in-flight requests; press Ctrl-C again to exit now."})
		stop()
		<-signals
		os.Exit(130)
	}()
	return runRentHost(ctx, cfg)
}

// rentTargetGet sends a model-discovery GET to the target, authenticated with
// the discovery key (else the inference key).
func rentTargetGet(cfg rentHostConfig, path string) (*http.Response, context.CancelFunc, error) {
	ctx, cancel := context.WithTimeout(context.Background(), 10*time.Second)
	req, err := http.NewRequestWithContext(ctx, http.MethodGet, cfg.Target+path, nil)
	if err != nil {
		cancel()
		return nil, nil, err
	}
	if key := firstNonEmpty(cfg.DiscoveryKey, cfg.TargetKey); key != "" {
		req.Header.Set("Authorization", "Bearer "+key)
	}
	res, err := cfg.Client.Do(req)
	if err != nil {
		cancel()
		return nil, nil, err
	}
	return res, cancel, nil
}

// rentTargetModels picks the models to advertise when --model is omitted. A
// emx gateway starts any registered model on demand, so for one only
// the models running now are advertised; other targets advertise /v1/models.
func rentTargetModels(cfg rentHostConfig) ([]string, error) {
	// Right after a load, emx's health probe can briefly report an
	// instance as unhealthy; wait a little for it instead of failing.
	deadline := time.Now().Add(rentDiscoveryWait)
	for {
		models, isEMX, transient, err := rentEMXRunningModels(cfg)
		if isEMX && transient && err != nil && time.Now().Before(deadline) {
			time.Sleep(time.Second)
			continue
		}
		if isEMX {
			return models, err
		}
		break
	}
	hints := []string{"Start the local gateway (lmx gateway start) and load a model (lmx model load <name>), or pass --target <base-url>.", "Or name the served models explicitly with --model <name>."}
	res, cancel, err := rentTargetGet(cfg, "/v1/models")
	if err != nil {
		return nil, cliError{"target_unreachable", "Could not list models at " + cfg.Target + "/v1/models: " + err.Error(), hints, nil}
	}
	defer cancel()
	defer res.Body.Close()
	var listing struct {
		Data []struct {
			ID string `json:"id"`
		} `json:"data"`
	}
	if res.StatusCode != http.StatusOK || json.NewDecoder(io.LimitReader(res.Body, 4<<20)).Decode(&listing) != nil {
		return nil, cliError{"target_models_failed", fmt.Sprintf("%s/v1/models returned HTTP %d.", cfg.Target, res.StatusCode), hints, nil}
	}
	models := []string{}
	for _, model := range listing.Data {
		if model.ID != "" && !sliceContainsString(models, model.ID) {
			models = append(models, model.ID)
		}
	}
	if len(models) == 0 {
		return nil, cliError{"no_models", cfg.Target + " serves no models.", hints, nil}
	}
	return models, nil
}

// rentDiscoveryWait bounds how long discovery waits for unhealthy instances.
var rentDiscoveryWait = 20 * time.Second

// rentEMXRunningModels lists the names of ready or starting instances
// when the target answers emx's management API (isEMX); transient
// reports instances whose health probe currently fails.
func rentEMXRunningModels(cfg rentHostConfig) (models []string, isEMX, transient bool, err error) {
	res, cancel, err := rentTargetGet(cfg, "/emx/v1/instances")
	if err != nil {
		return nil, false, false, nil
	}
	defer cancel()
	defer res.Body.Close()
	switch res.StatusCode {
	case http.StatusOK:
	case http.StatusUnauthorized, http.StatusForbidden:
		return nil, true, false, cliError{"target_models_failed", "The emx gateway at " + cfg.Target + " needs an admin key to list its running models.", []string{
			"export EMX_API_KEY=<admin key> and retry.",
			"Or name the models to rent out with --model <name>.",
		}, nil}
	default:
		return nil, false, false, nil
	}
	var instances []struct {
		Instance struct {
			Name string `json:"name"`
		} `json:"instance"`
		Status string `json:"status"`
	}
	if json.NewDecoder(io.LimitReader(res.Body, 4<<20)).Decode(&instances) != nil {
		return nil, false, false, nil
	}
	models = []string{}
	for _, entry := range instances {
		running := entry.Status == "ready" || entry.Status == "starting"
		transient = transient || entry.Status == "unhealthy"
		if running && entry.Instance.Name != "" && !sliceContainsString(models, entry.Instance.Name) {
			models = append(models, entry.Instance.Name)
		}
	}
	if len(models) == 0 {
		return nil, true, transient, cliError{"no_running_models", "No model is running on the emx gateway at " + cfg.Target + "; registered models are not rented out until loaded.", []string{
			"Load one first: lmx model load <name>.",
			"Or name the models to rent out with --model <name> (the gateway starts them on the first rental request).",
		}, nil}
	}
	return models, true, false, nil
}

// errRelayDrain asks for a prompt reconnect after the relay drained us.
var errRelayDrain = errors.New("relay is draining connections")

// runRentHost keeps the agent connected until ctx is cancelled (then drains)
// or the relay refuses it for good (a terminal cliError).
func runRentHost(ctx context.Context, cfg rentHostConfig) error {
	if cfg.BackoffBase <= 0 {
		cfg.BackoffBase = time.Second
	}
	if cfg.BackoffMax <= 0 {
		cfg.BackoffMax = 60 * time.Second
	}
	if cfg.DrainTimeout <= 0 {
		cfg.DrainTimeout = 2 * time.Minute
	}
	if cfg.Client == nil {
		cfg.Client = &http.Client{}
	}
	if cfg.Status == nil {
		cfg.Status = func(string, map[string]any) {}
	}
	attempt := 0
	for {
		connected, err := rentHostSession(ctx, cfg)
		if ctx.Err() != nil {
			cfg.Status("rent_host_stopped", nil)
			return nil
		}
		var terminal cliError
		if errors.As(err, &terminal) {
			return err
		}
		if connected {
			attempt = 0
		}
		delay := rentBackoff(cfg.BackoffBase, cfg.BackoffMax, attempt)
		if errors.Is(err, errRelayDrain) {
			delay = cfg.BackoffBase
		}
		attempt++
		reason := "connection closed"
		if err != nil {
			reason = err.Error()
		}
		cfg.Status("rent_host_disconnected", map[string]any{"reason": reason, "retryInSeconds": roundMetric(delay.Seconds())})
		select {
		case <-ctx.Done():
			cfg.Status("rent_host_stopped", nil)
			return nil
		case <-time.After(delay):
		}
	}
}

// rentBackoff is exponential with jitter in [d/2, d], capped at max.
func rentBackoff(base, maxDelay time.Duration, attempt int) time.Duration {
	delay := base
	for i := 0; i < attempt && delay < maxDelay; i++ {
		delay *= 2
	}
	if delay > maxDelay {
		delay = maxDelay
	}
	return delay/2 + time.Duration(rand.Int64N(int64(delay/2)+1))
}

// rentTerminalError turns a relay refusal into a cliError with a fix.
func rentTerminalError(cfg rentHostConfig, code, message string) error {
	site := strings.TrimRight(firstNonEmpty(cfg.SiteURL, defaultAPIURL), "/")
	hints := map[string][]string{
		relayproto.CodeProRequired:      {"Hosting rentals needs LocalMaxxing Pro: " + site + "/pro"},
		relayproto.CodeUnauthorized:     {"Run lmx auth login, lmx auth --key bhk_..., or set LMX_API_KEY."},
		relayproto.CodeListingNotOwned:  {"Host only your own listings: lmx rent list --mine."},
		relayproto.CodeListingNotFound:  {"Check the IDs with lmx rent list --mine, or create a listing at " + site + "/rentals/new."},
		relayproto.CodeProtoUnsupported: {"Update the CLI: lmx update."},
		relayproto.CodeRevoked:          {"Use a current API key (lmx auth login, lmx auth --key bhk_..., or LMX_API_KEY) on an account with LocalMaxxing Pro: " + site + "/pro"},
	}[code]
	if message == "" {
		message = "The relay refused this host (" + code + ")."
	}
	return cliError{code, message, hints, map[string]any{"listings": cfg.Listings}}
}

// rentHostSession runs one relay connection. connected reports whether the
// relay welcomed us (resets backoff).
func rentHostSession(ctx context.Context, cfg rentHostConfig) (connected bool, err error) {
	// The connection outlives ctx: cancelling ctx only starts the drain.
	connCtx, closeConn := context.WithCancel(context.Background())
	defer closeConn()
	dialTimer := time.AfterFunc(20*time.Second, closeConn)
	stopDialOnCancel := context.AfterFunc(ctx, closeConn)
	ws, res, err := websocket.Dial(connCtx, cfg.RelayURL, &websocket.DialOptions{
		Subprotocols: []string{relayproto.Subprotocol},
		HTTPHeader:   http.Header{"Authorization": {"Bearer " + cfg.APIKey}},
	})
	dialTimer.Stop()
	stopDialOnCancel()
	if err != nil {
		if res != nil && res.StatusCode == http.StatusUnauthorized {
			return false, rentTerminalError(cfg, relayproto.CodeUnauthorized, "The relay rejected the API key.")
		}
		return false, fmt.Errorf("could not reach relay %s: %w", cfg.RelayURL, err)
	}
	if ws.Subprotocol() != relayproto.Subprotocol {
		_ = ws.Close(websocket.StatusPolicyViolation, "subprotocol mismatch")
		return false, errors.New("relay did not negotiate " + relayproto.Subprotocol)
	}
	session, err := yamux.Server(relayproto.NetConn(connCtx, ws), relayproto.YamuxConfig())
	if err != nil {
		return false, err
	}
	defer session.Close()
	stream, err := session.OpenStream()
	if err != nil {
		return false, fmt.Errorf("control stream: %w", err)
	}
	control := relayproto.NewConn(stream)
	hello := relayproto.Hello{Type: relayproto.TypeHello, Proto: relayproto.Proto, LmxVersion: version, Listings: cfg.Listings, Models: cfg.Models, MaxConcurrency: cfg.MaxConcurrency}
	if err := control.Write(hello, 20*time.Second); err != nil {
		return false, fmt.Errorf("hello: %w", err)
	}
	kind, raw, err := control.Read(30 * time.Second)
	if err != nil {
		return false, fmt.Errorf("no welcome from relay: %w", err)
	}
	switch kind {
	case relayproto.TypeWelcome:
	case relayproto.TypeError:
		var notice relayproto.Notice
		_ = json.Unmarshal(raw, &notice)
		if relayproto.Terminal(notice.Code) {
			return false, rentTerminalError(cfg, notice.Code, notice.Message)
		}
		return false, fmt.Errorf("relay refused (%s): %s", notice.Code, notice.Message)
	default:
		return false, errors.New("unexpected control message " + kind)
	}
	var welcome relayproto.Welcome
	if err := json.Unmarshal(raw, &welcome); err != nil {
		return false, errors.New("malformed welcome")
	}
	heartbeat := time.Duration(max(welcome.HeartbeatSec, 1)) * time.Second
	statuses := []string{}
	inactive := []string{}
	for _, listing := range welcome.Listings {
		statuses = append(statuses, listing.ID+" ("+listing.Status+")")
		if listing.Status != "ACTIVE" {
			inactive = append(inactive, listing.ID)
		}
	}
	fields := map[string]any{"agent": welcome.AgentID, "listings": strings.Join(statuses, ", "), "models": strings.Join(cfg.Models, ", "), "target": cfg.Target, "maxConcurrency": cfg.MaxConcurrency}
	if len(inactive) > 0 {
		fields["note"] = "Renters reach only ACTIVE listings; activate " + strings.Join(inactive, ", ") + " on " + strings.TrimRight(cfg.SiteURL, "/") + "/rentals."
	}
	cfg.Status("rent_host_connected", fields)

	host := &rentHost{cfg: cfg, slots: make(chan struct{}, cfg.MaxConcurrency), idle: make(chan struct{}, 1)}
	go func() {
		for {
			stream, err := session.AcceptStream()
			if err != nil {
				return
			}
			host.inFlight.Add(1)
			go host.serve(stream)
		}
	}()

	type event struct {
		kind   string
		notice relayproto.Notice
	}
	events := make(chan event, 4)
	sessionDone := make(chan struct{})
	defer close(sessionDone)
	emit := func(ev event) {
		select {
		case events <- ev:
		case <-sessionDone:
		}
	}
	var lastSeen atomic.Int64
	lastSeen.Store(time.Now().UnixNano())
	go func() {
		for {
			kind, raw, err := control.Read(0)
			if err != nil {
				emit(event{kind: "closed"})
				return
			}
			lastSeen.Store(time.Now().UnixNano())
			var notice relayproto.Notice
			_ = json.Unmarshal(raw, &notice)
			switch kind {
			case relayproto.TypePing:
				_ = control.Write(relayproto.Notice{Type: relayproto.TypePong}, 10*time.Second)
			case relayproto.TypeDrain, relayproto.TypeRevoked, relayproto.TypeError:
				emit(event{kind: kind, notice: notice})
			}
		}
	}()

	ticker := time.NewTicker(heartbeat)
	defer ticker.Stop()
	var drainDeadline <-chan time.Time
	draining := ""
	startDrain := func(reason string) {
		if draining != "" {
			return
		}
		draining = reason
		host.draining.Store(true)
		drainDeadline = time.After(cfg.DrainTimeout)
		cfg.Status("rent_host_draining", map[string]any{"reason": reason, "inFlight": host.inFlight.Load()})
	}
	finishDrain := func() (bool, error) {
		if draining == "user" {
			return true, nil
		}
		return true, errRelayDrain
	}
	ctxDone := ctx.Done()
	for {
		if draining != "" && host.inFlight.Load() == 0 {
			return finishDrain()
		}
		select {
		case <-ctxDone:
			ctxDone = nil
			// Ask the relay to route nothing new here, then finish in-flight work.
			_ = control.Write(relayproto.Notice{Type: relayproto.TypeDrain, Reason: "host stopping"}, 5*time.Second)
			startDrain("user")
		case <-host.idle:
		case <-drainDeadline:
			cfg.Status("rent_host_drain_timeout", map[string]any{"abandoned": host.inFlight.Load()})
			return finishDrain()
		case ev := <-events:
			switch ev.kind {
			case relayproto.TypeDrain:
				startDrain(firstNonEmpty(ev.notice.Reason, "relay"))
			case relayproto.TypeRevoked, relayproto.TypeError:
				if relayproto.Terminal(ev.notice.Code) {
					return true, rentTerminalError(cfg, ev.notice.Code, ev.notice.Message)
				}
				return true, fmt.Errorf("relay closed the connection (%s): %s", ev.notice.Code, ev.notice.Message)
			default:
				if draining == "user" {
					return true, nil
				}
				return true, errors.New("relay connection closed")
			}
		case <-session.CloseChan():
			if draining == "user" {
				return true, nil
			}
			return true, errors.New("relay connection lost")
		case <-ticker.C:
			if time.Since(time.Unix(0, lastSeen.Load())) > relayproto.MissedHeartbeats*heartbeat {
				return true, errors.New("relay heartbeat timeout")
			}
			if err := control.Write(relayproto.Notice{Type: relayproto.TypePing}, 10*time.Second); err != nil {
				return true, errors.New("relay heartbeat failed")
			}
		}
	}
}

// rentHost serves request streams for one relay connection.
type rentHost struct {
	cfg      rentHostConfig
	slots    chan struct{}
	inFlight atomic.Int64
	idle     chan struct{}
	draining atomic.Bool
}

// serve handles one request stream: read the request, enforce the allowlist,
// model list and concurrency, proxy to the target, stream the response back.
func (h *rentHost) serve(stream net.Conn) {
	defer func() {
		_ = stream.Close()
		if h.inFlight.Add(-1) == 0 {
			select {
			case h.idle <- struct{}{}:
			default:
			}
		}
	}()
	started := time.Now()
	_ = stream.SetReadDeadline(time.Now().Add(30 * time.Second))
	reader := bufio.NewReader(stream)
	req, err := http.ReadRequest(reader)
	if err != nil {
		return
	}
	body, err := io.ReadAll(io.LimitReader(req.Body, relayproto.MaxRequestBody+1))
	_ = req.Body.Close()
	_ = stream.SetReadDeadline(time.Time{})
	out := deadlineWriter{stream}
	if err != nil {
		writeRentError(out, http.StatusBadRequest, "invalid_request", "could not read the request body")
		return
	}
	status, model := h.handle(out, reader, req, body)
	h.cfg.Status("rent_request", map[string]any{
		"listing": req.Header.Get(relayproto.HeaderListing), "model": model, "path": req.Method + " " + req.URL.Path,
		"status": status, "durationMs": time.Since(started).Milliseconds(),
	})
}

// handle returns the response status (499 when the renter cancelled) and model.
func (h *rentHost) handle(out io.Writer, reader *bufio.Reader, req *http.Request, body []byte) (int, string) {
	if h.draining.Load() {
		return writeRentError(out, http.StatusServiceUnavailable, "agent_draining", "this host is shutting down"), ""
	}
	if !relayproto.AllowedRoute(req.Method, req.URL.Path) {
		return writeRentError(out, http.StatusNotFound, "not_found", "rentals serve only POST /v1/chat/completions, /v1/completions, /v1/embeddings and GET /v1/models"), ""
	}
	if req.URL.Path == "/v1/models" {
		return writeRentModels(out, h.cfg.Models), ""
	}
	if len(body) > relayproto.MaxRequestBody {
		return writeRentError(out, http.StatusRequestEntityTooLarge, "body_too_large", "request body is too large"), ""
	}
	canonical, model, err := rentCanonicalBody(body)
	if err != nil {
		return writeRentError(out, http.StatusBadRequest, "invalid_request", err.Error()), ""
	}
	if !sliceContainsString(h.cfg.Models, model) {
		return writeRentError(out, http.StatusBadRequest, "model_not_available", fmt.Sprintf("model %q is not served by this host; available: %s", model, strings.Join(h.cfg.Models, ", "))), model
	}
	select {
	case h.slots <- struct{}{}:
		defer func() { <-h.slots }()
	default:
		return writeRentError(out, http.StatusTooManyRequests, "rate_limited", fmt.Sprintf("this host is at its concurrency limit (%d)", h.cfg.MaxConcurrency)), model
	}

	// Closing the stream (relay or renter gone) cancels the upstream request.
	ctx, cancel := context.WithCancel(context.Background())
	defer cancel()
	go func() {
		var b [1]byte
		_, _ = reader.Read(b[:])
		cancel()
	}()
	upstream, err := http.NewRequestWithContext(ctx, req.Method, h.cfg.Target+req.URL.Path, bytes.NewReader(canonical))
	if err != nil {
		return writeRentError(out, http.StatusBadGateway, "target_failed", "could not build the upstream request"), model
	}
	for _, name := range []string{"Content-Type", "Accept", "User-Agent"} {
		if value := req.Header.Get(name); value != "" {
			upstream.Header.Set(name, value)
		}
	}
	if h.cfg.TargetKey != "" {
		upstream.Header.Set("Authorization", "Bearer "+h.cfg.TargetKey)
	}
	res, err := h.cfg.Client.Do(upstream)
	if err != nil {
		if ctx.Err() != nil {
			return 499, model
		}
		return writeRentError(out, http.StatusBadGateway, "target_unreachable", "the host's inference server did not answer"), model
	}
	defer res.Body.Close()
	response := &http.Response{
		StatusCode: res.StatusCode, ProtoMajor: 1, ProtoMinor: 1,
		Header: http.Header{}, Body: res.Body, ContentLength: res.ContentLength,
	}
	if contentType := res.Header.Get("Content-Type"); contentType != "" {
		response.Header.Set("Content-Type", contentType)
	}
	if response.ContentLength < 0 {
		response.TransferEncoding = []string{"chunked"}
	}
	if err := response.Write(out); err != nil && ctx.Err() != nil {
		return 499, model
	}
	return res.StatusCode, model
}

// rentCanonicalBody parses a JSON object request body and returns it
// re-encoded, with the model it names. Duplicate keys and case variants of
// "model" are refused: a target whose parser takes the first or the exact
// key would otherwise run a different model than the one checked here.
func rentCanonicalBody(body []byte) ([]byte, string, error) {
	invalid := errors.New("request body must be a JSON object")
	decoder := json.NewDecoder(bytes.NewReader(body))
	if token, err := decoder.Token(); err != nil || token != json.Delim('{') {
		return nil, "", invalid
	}
	var out bytes.Buffer
	out.Grow(len(body))
	out.WriteByte('{')
	seen := map[string]bool{}
	model := ""
	for decoder.More() {
		token, err := decoder.Token()
		key, ok := token.(string)
		if err != nil || !ok {
			return nil, "", invalid
		}
		if seen[key] {
			return nil, "", fmt.Errorf("duplicate key %q in the request body", key)
		}
		if key != "model" && strings.EqualFold(key, "model") {
			return nil, "", fmt.Errorf("key %q: the model must be sent as \"model\"", key)
		}
		seen[key] = true
		var value json.RawMessage
		if err := decoder.Decode(&value); err != nil {
			return nil, "", invalid
		}
		if key == "model" && json.Unmarshal(value, &model) != nil {
			return nil, "", errors.New("model must be a string")
		}
		if len(seen) > 1 {
			out.WriteByte(',')
		}
		encodedKey, _ := json.Marshal(key)
		out.Write(encodedKey)
		out.WriteByte(':')
		out.Write(value)
	}
	if token, err := decoder.Token(); err != nil || token != json.Delim('}') {
		return nil, "", invalid
	}
	if _, err := decoder.Token(); err != io.EOF {
		return nil, "", invalid
	}
	out.WriteByte('}')
	return out.Bytes(), model, nil
}

// deadlineWriter bounds each write so a stalled relay cannot pin a request.
type deadlineWriter struct{ conn net.Conn }

func (w deadlineWriter) Write(p []byte) (int, error) {
	_ = w.conn.SetWriteDeadline(time.Now().Add(30 * time.Second))
	return w.conn.Write(p)
}

func writeRentJSON(out io.Writer, status int, body []byte) int {
	response := &http.Response{
		StatusCode: status, ProtoMajor: 1, ProtoMinor: 1,
		Header:        http.Header{"Content-Type": {"application/json"}},
		Body:          io.NopCloser(bytes.NewReader(body)),
		ContentLength: int64(len(body)),
	}
	_ = response.Write(out)
	return status
}

func writeRentError(out io.Writer, status int, code, message string) int {
	return writeRentJSON(out, status, relayproto.ErrorJSON(code, message))
}

func writeRentModels(out io.Writer, models []string) int {
	data := make([]map[string]any, 0, len(models))
	for _, model := range models {
		data = append(data, map[string]any{"id": model, "object": "model", "owned_by": "localmaxxing-rental"})
	}
	body, _ := json.Marshal(map[string]any{"object": "list", "data": data})
	return writeRentJSON(out, http.StatusOK, body)
}

// ── rent connect ────────────────────────────────────────────────────────────

func handleRentConnect(args cliArgs) error {
	listing := positional(args, 2)
	if listing == "" {
		return cliError{"missing_listing", "lmx rent connect needs a listing ID.", []string{"Browse listings with lmx rent list, then run lmx rent connect <listingId>."}, nil}
	}
	key := apiKey(args)
	if key == "" {
		return missingAPIKey("lmx rent connect authenticates rental requests with your LocalMaxxing API key.")
	}
	port, err := intOption(args, 8000, 0, "port")
	if err != nil {
		return err
	}
	host := strings.Trim(firstNonEmpty(opt(args, "host"), "127.0.0.1"), "[]")
	if addr, err := netip.ParseAddr(host); err == nil && addr.IsUnspecified() {
		return cliError{"invalid_option", "--host " + host + " listens on every interface; requests could not be checked against DNS rebinding.", []string{"Pass one specific address, e.g. --host 192.168.1.20, or keep the default 127.0.0.1."}, nil}
	}
	address := net.JoinHostPort(host, strconv.Itoa(port))
	listener, err := net.Listen("tcp", address)
	if err != nil {
		return cliError{"listen_failed", "Could not listen on " + address + ": " + err.Error(), []string{"Pick a free port with --port <n>."}, nil}
	}
	port = listener.Addr().(*net.TCPAddr).Port
	if !isLoopbackHost(host) {
		printStatus(args, "rent_connect_exposed_warning", map[string]any{
			"warning": "--host " + host + " is not loopback: anyone who can reach " + net.JoinHostPort(host, strconv.Itoa(port)) + " sends rental requests with YOUR API key and spends YOUR credits. Use only on a trusted network.",
		})
	}
	upstream := apiURL(args) + "/api/rentals/" + url.PathEscape(listing)
	handler := newRentConnectHandler(upstream, key, opt(args, "rental-token"), rentConnectHosts(host, port), func(event string, fields map[string]any) { printStatus(args, event, fields) })
	server := &http.Server{Handler: handler, ReadHeaderTimeout: 15 * time.Second}
	baseURL := "http://" + net.JoinHostPort(host, strconv.Itoa(port)) + "/v1"
	printInfo(args, "rent_connect_ready", map[string]any{"baseUrl": baseURL, "listing": listing, "upstream": upstream + "/v1", "next": "Point any OpenAI-compatible client at " + baseURL + " (API key: any value). Browser requests (with an Origin header) are refused. Ctrl-C stops."})
	if hasFlag(args, "json") {
		printJSON(args, map[string]any{"event": "rent_connect_ready", "baseUrl": baseURL, "listing": listing})
	}

	signals := make(chan os.Signal, 1)
	signal.Notify(signals, os.Interrupt, syscall.SIGTERM)
	defer signal.Stop(signals)
	served := make(chan error, 1)
	go func() { served <- server.Serve(listener) }()
	select {
	case err := <-served:
		if !errors.Is(err, http.ErrServerClosed) {
			return err
		}
		return nil
	case <-signals:
		ctx, cancel := context.WithTimeout(context.Background(), 10*time.Second)
		defer cancel()
		_ = server.Shutdown(ctx)
		printStatus(args, "rent_connect_stopped", nil)
		return nil
	}
}

// rentConnectHosts are the Host header values the local proxy accepts: the
// loopback names plus the configured address. Any other Host (a DNS-rebound
// attacker domain) is refused.
func rentConnectHosts(host string, port int) []string {
	p := strconv.Itoa(port)
	return []string{"127.0.0.1:" + p, "localhost:" + p, "[::1]:" + p, strings.ToLower(net.JoinHostPort(host, p))}
}

// newRentConnectHandler proxies the four rental routes to <upstream>/v1/...
// with the renter's credentials, streaming responses as they arrive. It
// attaches the API key, so it accepts only exact routes, allowed Host
// headers and no browser (Origin) requests.
func newRentConnectHandler(upstream, key, rentalToken string, allowedHosts []string, status func(string, map[string]any)) http.Handler {
	client := &http.Client{}
	return http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		started := time.Now()
		if !sliceContainsString(allowedHosts, strings.ToLower(r.Host)) {
			relayproto.WriteError(w, http.StatusForbidden, "host_not_allowed", "Host "+r.Host+" is not allowed; use "+allowedHosts[0])
			return
		}
		if _, browser := r.Header["Origin"]; browser {
			relayproto.WriteError(w, http.StatusForbidden, "origin_not_allowed", "browser requests are not accepted by lmx rent connect")
			return
		}
		// Matched on the raw path, so %2F and dot-segments never pass; the
		// outbound path is then exactly one of the allowlisted constants.
		path := r.URL.EscapedPath()
		if !relayproto.AllowedRoute(r.Method, path) {
			relayproto.WriteError(w, http.StatusNotFound, "not_found", "only POST /v1/chat/completions, /v1/completions, /v1/embeddings and GET /v1/models are proxied")
			return
		}
		target := upstream + path
		if r.URL.RawQuery != "" {
			target += "?" + r.URL.RawQuery
		}
		req, err := http.NewRequestWithContext(r.Context(), r.Method, target, r.Body)
		if err != nil {
			relayproto.WriteError(w, http.StatusBadRequest, "bad_request", err.Error())
			return
		}
		req.ContentLength = r.ContentLength
		for _, name := range []string{"Content-Type", "Accept", "User-Agent"} {
			if value := r.Header.Get(name); value != "" {
				req.Header.Set(name, value)
			}
		}
		req.Header.Set("Authorization", "Bearer "+key)
		if rentalToken != "" {
			req.Header.Set("X-Rental-Token", rentalToken)
		}
		res, err := client.Do(req)
		if err != nil {
			if r.Context().Err() == nil {
				relayproto.WriteError(w, http.StatusBadGateway, "upstream_unreachable", "LocalMaxxing did not answer: "+err.Error())
			}
			return
		}
		defer res.Body.Close()
		for name, values := range res.Header {
			switch http.CanonicalHeaderKey(name) {
			case "Connection", "Keep-Alive", "Transfer-Encoding", "Content-Length", "Content-Encoding":
				continue
			}
			w.Header()[name] = values
		}
		w.WriteHeader(res.StatusCode)
		controller := http.NewResponseController(w)
		buf := make([]byte, 32<<10)
		for {
			n, readErr := res.Body.Read(buf)
			if n > 0 {
				if _, err := w.Write(buf[:n]); err != nil {
					break
				}
				_ = controller.Flush()
			}
			if readErr != nil {
				break
			}
		}
		if status != nil {
			status("rent_connect_request", map[string]any{"path": r.Method + " " + r.URL.Path, "status": res.StatusCode, "durationMs": time.Since(started).Milliseconds()})
		}
	})
}

// ── rent list ───────────────────────────────────────────────────────────────

func handleRentList(args cliArgs) error {
	mine := hasFlag(args, "mine")
	path, key := "/api/rentals", ""
	if mine {
		path = "/api/rentals/mine"
		if key = apiKey(args); key == "" {
			return missingAPIKey("lmx rent list --mine lists your own listings and needs your API key.")
		}
	}
	parsed, err := fetchJSON("GET", apiURL(args)+path, key, nil)
	if err != nil {
		return err
	}
	listings := []map[string]any{}
	for _, item := range anySlice(asObject(parsed)["listings"]) {
		if listing := asObject(item); listing != nil {
			listings = append(listings, listing)
		}
	}
	if hasFlag(args, "json") && opt(args, "out") == "" {
		printJSON(args, map[string]any{"listings": listings})
		return nil
	}
	if opt(args, "out") != "" {
		return writeOrPrintJSON("rentals", args, map[string]any{"listings": listings})
	}
	if len(listings) == 0 {
		next := "No active listings right now."
		if mine {
			next = "Create a listing at " + apiURL(args) + "/rentals/new, then run lmx rent host <listingId>."
		}
		printInfo(args, "no_rentals", map[string]any{"next": next})
		return nil
	}
	headers := []string{"ID", "TITLE", "MODEL", "IN/1K", "CACHED/1K", "OUT/1K", "SESSIONS"}
	if mine {
		headers = append([]string{headers[0], "STATUS"}, headers[1:]...)
	} else {
		headers = append(headers, "HARDWARE", "OWNER")
	}
	rows := [][]string{}
	for _, listing := range listings {
		model := firstNonEmpty(stringValue(asObject(listing["model"])["hfId"]), stringValue(listing["modelLabel"]), "-")
		// Credits per 1k tokens; a null cached-input price means the input price applies.
		input := numberField(listing, "pricePer1kInputTokens")
		cached := input
		if value, ok := listing["pricePer1kCachedInputTokens"].(float64); ok {
			cached = value
		}
		output := numberField(listing, "pricePer1kOutputTokens")
		sessions := "0"
		if value, ok := asObject(listing["_count"])["sessions"].(float64); ok {
			sessions = strconv.Itoa(int(value))
		}
		row := []string{
			stringValue(listing["id"]), truncateRentCell(stringValue(listing["title"]), 40), model,
			strconv.FormatFloat(input, 'f', -1, 64), strconv.FormatFloat(cached, 'f', -1, 64), strconv.FormatFloat(output, 'f', -1, 64),
			sessions,
		}
		if mine {
			row = append([]string{row[0], stringValue(listing["status"])}, row[1:]...)
		} else {
			hardware := firstNonEmpty(stringValue(listing["hardwareLabel"]), stringValue(asObject(listing["hardware"])["gpuName"]), "-")
			owner := firstNonEmpty(stringValue(asObject(listing["owner"])["username"]), stringValue(asObject(listing["owner"])["name"]), "-")
			row = append(row, truncateRentCell(hardware, 30), owner)
		}
		rows = append(rows, row)
	}
	fmt.Print(asciiTable(headers, rows))
	return nil
}

func truncateRentCell(value string, limit int) string {
	if len([]rune(value)) <= limit {
		return value
	}
	return string([]rune(value)[:limit-1]) + "…"
}
