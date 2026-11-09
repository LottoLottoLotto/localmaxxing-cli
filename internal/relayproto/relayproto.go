// Package relayproto is the wire contract between `lmx rent host` (the agent
// on an owner's machine) and lmx-relay: control messages, their JSON line
// codec, the WebSocket transport adapter and the request-route allowlist.
//
// Transport: the agent dials GET /relay/v1/agent as a WebSocket with
// subprotocol lmx-relay.v1. Binary messages carry one yamux session; the
// relay is the yamux client and the agent the yamux server. The agent opens
// the first stream itself as the control stream (newline-delimited JSON);
// every later stream is opened by the relay and carries exactly one renter
// request as plain HTTP/1.1. Closing a request stream early cancels it.
package relayproto

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
	"sync"
	"time"

	"github.com/coder/websocket"
	"github.com/hashicorp/yamux"
)

const (
	// Proto is the control protocol version carried in hello.
	Proto = 1
	// Subprotocol is the WebSocket subprotocol both sides must negotiate.
	Subprotocol = "lmx-relay.v1"
	// AgentPath is the public WebSocket endpoint on the relay.
	AgentPath = "/relay/v1/agent"
	// DefaultHeartbeat is the ping interval the relay announces in welcome.
	DefaultHeartbeat = 20 * time.Second
	// MissedHeartbeats is how many silent intervals close a connection.
	MissedHeartbeats = 3
	// MaxControlLine bounds one control message.
	MaxControlLine = 64 << 10
	// MaxRequestBody bounds one renter request body on every hop.
	MaxRequestBody = 16 << 20
	// wsReadLimit bounds one WebSocket message; it must exceed the largest
	// yamux frame (the 256 KiB stream window plus its header).
	wsReadLimit = 1 << 20
)

// Request headers the relay adds to every request stream.
const (
	HeaderListing = "X-Lmx-Listing"
	HeaderSession = "X-Lmx-Session"
	HeaderMethod  = "X-Lmx-Method"
	HeaderPath    = "X-Lmx-Path"
)

// HeaderRelayError marks a relay-originated error response with its code.
// The relay never copies agent headers, so only the relay can set it.
const HeaderRelayError = "X-Lmx-Relay-Error"

// Control message types.
const (
	TypeHello   = "hello"
	TypeWelcome = "welcome"
	TypeError   = "error"
	TypePing    = "ping"
	TypePong    = "pong"
	TypeDrain   = "drain"
	TypeRevoked = "revoked"
)

// Error codes carried by error/revoked messages and the authorize endpoint.
const (
	CodeProRequired      = "pro_required"
	CodeUnauthorized     = "unauthorized"
	CodeListingNotOwned  = "listing_not_owned"
	CodeListingNotFound  = "listing_not_found"
	CodeProtoUnsupported = "proto_unsupported"
	// CodeSiteUnavailable is a transient authorize failure; agents retry it.
	CodeSiteUnavailable = "site_unavailable"
	// CodeRelayDraining rejects new agents while the relay shuts down.
	CodeRelayDraining = "relay_draining"
	CodeBadHello      = "bad_hello"
	// CodeTooManyAgents: a relay, per-IP or per-API-key agent limit is reached.
	CodeTooManyAgents = "too_many_agents"
	// CodeRateLimited: too many agent connection attempts from one address.
	CodeRateLimited = "rate_limited"
	// CodeRevoked: the API key or the account's hosting was revoked while
	// the agent was connecting.
	CodeRevoked = "revoked"
)

// Terminal reports whether an error/revoked code must stop an agent instead
// of reconnecting: retrying cannot succeed until the owner changes something.
func Terminal(code string) bool {
	switch code {
	case CodeProRequired, CodeUnauthorized, CodeListingNotOwned, CodeListingNotFound, CodeProtoUnsupported, CodeRevoked:
		return true
	}
	return false
}

// Envelope reads only the type of a control message.
type Envelope struct {
	Type string `json:"type"`
}

// Hello is the first control message, agent → relay.
type Hello struct {
	Type           string   `json:"type"`
	Proto          int      `json:"proto"`
	LmxVersion     string   `json:"lmxVersion"`
	Listings       []string `json:"listings"`
	Models         []string `json:"models"`
	MaxConcurrency int      `json:"maxConcurrency"`
}

// ListingStatus is one listing in welcome.
type ListingStatus struct {
	ID     string `json:"id"`
	Status string `json:"status"`
}

// Welcome accepts an agent, relay → agent.
type Welcome struct {
	Type         string          `json:"type"`
	AgentID      string          `json:"agentId"`
	HeartbeatSec int             `json:"heartbeatSec"`
	Listings     []ListingStatus `json:"listings"`
}

// Notice is error, drain and revoked (and ping/pong, which carry only a type).
type Notice struct {
	Type    string `json:"type"`
	Code    string `json:"code,omitempty"`
	Message string `json:"message,omitempty"`
	Reason  string `json:"reason,omitempty"`
}

// Conn is a newline-delimited JSON codec over a control stream. Writes are
// serialized; reads must come from one goroutine.
type Conn struct {
	conn    net.Conn
	reader  *bufio.Reader
	writeMu sync.Mutex
}

func NewConn(conn net.Conn) *Conn {
	return &Conn{conn: conn, reader: bufio.NewReaderSize(conn, 4096)}
}

// Write sends one message as a JSON line, bounded by timeout.
func (c *Conn) Write(message any, timeout time.Duration) error {
	data, err := json.Marshal(message)
	if err != nil {
		return err
	}
	data = append(data, '\n')
	c.writeMu.Lock()
	defer c.writeMu.Unlock()
	if timeout > 0 {
		_ = c.conn.SetWriteDeadline(time.Now().Add(timeout))
		defer c.conn.SetWriteDeadline(time.Time{})
	}
	_, err = c.conn.Write(data)
	return err
}

// Read returns the next message's type and raw JSON. A zero timeout waits
// indefinitely.
func (c *Conn) Read(timeout time.Duration) (string, []byte, error) {
	if timeout > 0 {
		_ = c.conn.SetReadDeadline(time.Now().Add(timeout))
		defer c.conn.SetReadDeadline(time.Time{})
	}
	var line []byte
	for {
		chunk, isPrefix, err := c.reader.ReadLine()
		if err != nil {
			return "", nil, err
		}
		line = append(line, chunk...)
		if len(line) > MaxControlLine {
			return "", nil, errors.New("control message too large")
		}
		if !isPrefix {
			break
		}
	}
	line = bytes.TrimSpace(line)
	var envelope Envelope
	if err := json.Unmarshal(line, &envelope); err != nil || envelope.Type == "" {
		return "", nil, fmt.Errorf("malformed control message")
	}
	return envelope.Type, line, nil
}

// Close closes the underlying stream.
func (c *Conn) Close() error { return c.conn.Close() }

// NetConn adapts a WebSocket to the byte stream yamux runs over, using binary
// messages. Cancelling ctx closes the connection. websocket.NetConn disables
// the read limit, so it is set again afterwards.
func NetConn(ctx context.Context, ws *websocket.Conn) net.Conn {
	conn := websocket.NetConn(ctx, ws, websocket.MessageBinary)
	ws.SetReadLimit(wsReadLimit)
	return conn
}

// YamuxConfig is the session configuration both sides use.
func YamuxConfig() *yamux.Config {
	config := yamux.DefaultConfig()
	config.LogOutput = io.Discard
	config.EnableKeepAlive = true
	config.KeepAliveInterval = 30 * time.Second
	config.ConnectionWriteTimeout = 30 * time.Second
	return config
}

// AllowedRoute is the only set of requests an agent ever forwards to its local
// target; anything else (including a target's admin API) is refused.
func AllowedRoute(method, path string) bool {
	switch path {
	case "/v1/chat/completions", "/v1/completions", "/v1/embeddings":
		return method == http.MethodPost
	case "/v1/models":
		return method == http.MethodGet
	}
	return false
}

// ErrorBody is the JSON error shape of the relay and agents.
type ErrorBody struct {
	Error ErrorDetail `json:"error"`
}

type ErrorDetail struct {
	Code    string `json:"code"`
	Message string `json:"message"`
}

// ErrorJSON encodes an error body.
func ErrorJSON(code, message string) []byte {
	data, _ := json.Marshal(ErrorBody{Error: ErrorDetail{Code: code, Message: message}})
	return data
}

// WriteError writes an error body to an HTTP response.
func WriteError(w http.ResponseWriter, status int, code, message string) {
	w.Header().Set("Content-Type", "application/json")
	w.WriteHeader(status)
	_, _ = w.Write(ErrorJSON(code, message))
}
