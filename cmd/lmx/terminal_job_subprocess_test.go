package main

import (
	"os"
	"testing"
)

const terminalJobTestHelperEnv = "LMX_TEST_TERMINAL_JOB_HELPER"

func TestMain(m *testing.M) {
	switch os.Getenv(terminalJobTestHelperEnv) {
	case "detached-output":
		_, _ = os.Stdout.WriteString("detached worker stdout\n")
		_, _ = os.Stderr.WriteString("detached worker event\n")
		os.Exit(0)
	case "wait-for-termination":
		ctx, stop := terminalSignalContext()
		defer stop()
		if err := os.WriteFile(os.Getenv("LMX_TEST_TERMINAL_READY"), []byte("ready\n"), 0o644); err != nil {
			os.Exit(2)
		}
		<-ctx.Done()
		if eventsPath := os.Getenv("LMX_TEST_TERMINAL_EVENTS"); eventsPath != "" {
			events, err := os.ReadFile(eventsPath)
			if err != nil {
				os.Exit(3)
			}
			if err := os.WriteFile(os.Getenv("LMX_TEST_TERMINAL_OBSERVED"), events, 0o644); err != nil {
				os.Exit(4)
			}
		}
		if err := os.WriteFile(os.Getenv("LMX_TEST_TERMINAL_SIGNALLED"), []byte("terminated\n"), 0o644); err != nil {
			os.Exit(5)
		}
		os.Exit(0)
	default:
		os.Exit(m.Run())
	}
}
