// Command lmx-relay carries LocalMaxxing rental traffic between the site and
// owner machines running `lmx rent host`. It is pure transport: no database,
// no billing. See internal/relay and the README "Rental relay" section.
//
// Environment:
//
//	LMX_RELAY_ADDR               public listener: agents + /healthz (default :8080)
//	LMX_RELAY_INTERNAL_ADDR      internal listener: /internal/* for the site (default :8081)
//	RELAY_INTERNAL_SECRET        site → relay bearer (required, >= 32 chars)
//	RELAY_AUTHORIZE_SECRET       relay → site bearer (required, >= 32 chars)
//	LMX_SITE_URL                 site origin for agent authorization (required)
//	LMX_RELAY_MAX_HANDSHAKES     unauthenticated agent connections (default 256)
//	LMX_RELAY_MAX_HANDSHAKES_PER_IP unauthenticated agent connections per client IP or IPv6 /64 (default 2)
//	LMX_RELAY_MAX_AGENTS         registered agents (default 2000)
//	LMX_RELAY_MAX_AGENTS_PER_IP  agent connections per client IP or IPv6 /64, handshakes included (default 20)
//	LMX_RELAY_MAX_AGENTS_PER_KEY registered agents per API key (default 8)
//	LMX_RELAY_MAX_AGENTS_PER_OWNER registered agents per account (default 32)
//	LMX_RELAY_CLIENT_IP_HEADER   client address header behind a CDN, e.g. CF-Connecting-IP (default: rightmost X-Forwarded-For)
//	LMX_RELAY_DRAIN_SECONDS      shutdown grace for in-flight requests (default 60)
package main

import (
	"context"
	"errors"
	"fmt"
	"log/slog"
	"net/http"
	"os"
	"os/signal"
	"strconv"
	"syscall"
	"time"

	"github.com/LottoLottoLotto/localmaxxing-cli/internal/relay"
)

func main() {
	logger := slog.New(slog.NewJSONHandler(os.Stdout, nil))
	fail := func(message string) {
		logger.Error(message)
		os.Exit(2)
	}
	publicAddr := envOr("LMX_RELAY_ADDR", ":8080")
	internalAddr := envOr("LMX_RELAY_INTERNAL_ADDR", ":8081")
	siteURL := os.Getenv("LMX_SITE_URL")
	drainSeconds, err := envInt("LMX_RELAY_DRAIN_SECONDS", 60, 0)
	if err != nil {
		fail(err.Error())
	}
	maxHandshakes, err := envInt("LMX_RELAY_MAX_HANDSHAKES", relay.DefaultMaxHandshakes, 1)
	if err != nil {
		fail(err.Error())
	}
	maxHandshakesPerIP, err := envInt("LMX_RELAY_MAX_HANDSHAKES_PER_IP", relay.DefaultMaxHandshakesPerIP, 1)
	if err != nil {
		fail(err.Error())
	}
	maxAgents, err := envInt("LMX_RELAY_MAX_AGENTS", relay.DefaultMaxAgents, 1)
	if err != nil {
		fail(err.Error())
	}
	maxPerIP, err := envInt("LMX_RELAY_MAX_AGENTS_PER_IP", relay.DefaultMaxAgentsPerIP, 1)
	if err != nil {
		fail(err.Error())
	}
	maxPerKey, err := envInt("LMX_RELAY_MAX_AGENTS_PER_KEY", relay.DefaultMaxAgentsPerKey, 1)
	if err != nil {
		fail(err.Error())
	}
	maxPerOwner, err := envInt("LMX_RELAY_MAX_AGENTS_PER_OWNER", relay.DefaultMaxAgentsPerOwner, 1)
	if err != nil {
		fail(err.Error())
	}
	clientIPHeader := os.Getenv("LMX_RELAY_CLIENT_IP_HEADER")

	server, err := relay.New(relay.Config{
		SiteURL:            siteURL,
		InternalSecret:     os.Getenv("RELAY_INTERNAL_SECRET"),
		AuthorizeSecret:    os.Getenv("RELAY_AUTHORIZE_SECRET"),
		MaxHandshakes:      maxHandshakes,
		MaxHandshakesPerIP: maxHandshakesPerIP,
		MaxAgents:          maxAgents,
		MaxAgentsPerIP:     maxPerIP,
		MaxAgentsPerKey:    maxPerKey,
		MaxAgentsPerOwner:  maxPerOwner,
		ClientIPHeader:     clientIPHeader,
		Logger:             logger,
	})
	if err != nil {
		fail(err.Error())
	}
	servers := []*http.Server{
		{Addr: publicAddr, Handler: server.PublicHandler(), ReadHeaderTimeout: 15 * time.Second, MaxHeaderBytes: 16 << 10},
		{Addr: internalAddr, Handler: server.InternalHandler(), ReadHeaderTimeout: 15 * time.Second},
	}
	stop := make(chan os.Signal, 1)
	signal.Notify(stop, syscall.SIGTERM, os.Interrupt)
	failed := make(chan error, len(servers))
	for _, httpServer := range servers {
		go func(httpServer *http.Server) {
			if err := httpServer.ListenAndServe(); err != nil && !errors.Is(err, http.ErrServerClosed) {
				failed <- fmt.Errorf("%s: %w", httpServer.Addr, err)
			}
		}(httpServer)
	}
	logger.Info("lmx-relay listening", "public", publicAddr, "internal", internalAddr, "site", siteURL,
		"maxHandshakes", maxHandshakes, "maxHandshakesPerIp", maxHandshakesPerIP, "maxAgents", maxAgents, "maxAgentsPerIp", maxPerIP, "maxAgentsPerKey", maxPerKey,
		"maxAgentsPerOwner", maxPerOwner, "clientIpHeader", clientIPHeader)
	select {
	case err := <-failed:
		logger.Error("listen failed", "error", err.Error())
		os.Exit(1)
	case sig := <-stop:
		logger.Info("draining", "signal", sig.String(), "graceSeconds", drainSeconds)
	}

	ctx, cancel := context.WithTimeout(context.Background(), time.Duration(drainSeconds)*time.Second)
	defer cancel()
	if err := server.Drain(ctx); err != nil {
		logger.Warn("drain grace expired with requests in flight")
	}
	server.Close()
	shutdownCtx, cancelShutdown := context.WithTimeout(context.Background(), 5*time.Second)
	defer cancelShutdown()
	for _, httpServer := range servers {
		_ = httpServer.Shutdown(shutdownCtx)
	}
	logger.Info("lmx-relay stopped")
}

func envOr(key, fallback string) string {
	if value := os.Getenv(key); value != "" {
		return value
	}
	return fallback
}

// envInt reads an integer >= minimum, or fallback when unset.
func envInt(key string, fallback, minimum int) (int, error) {
	value := os.Getenv(key)
	if value == "" {
		return fallback, nil
	}
	parsed, err := strconv.Atoi(value)
	if err != nil || parsed < minimum {
		return 0, fmt.Errorf("%s must be an integer >= %d", key, minimum)
	}
	return parsed, nil
}
