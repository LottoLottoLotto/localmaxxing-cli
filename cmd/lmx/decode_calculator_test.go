package main

import (
	"math"
	"testing"
)

func TestDecodeCalculatorMatchesWebsiteDenseVector(t *testing.T) {
	input := decodeCalculatorInput{
		Hardware: decodeHardwareInput{CapacityGB: 24, BandwidthGBps: 1000, OverheadGB: 1},
		Model:    decodeModelInput{Architecture: "dense", TotalParamsB: 7, ActiveParamsB: 7, WeightBits: 8, MoERouting: "minimum"},
		Cache:    decodeCacheInput{GlobalLayers: 1, KVHeads: 1, HeadDim: 1, StoreBits: 16, ReadBits: 16, SlidingWindow: 4096, FixedStateStoreGB: 1, FixedStateReadGB: 0.1},
		Workload: decodeWorkloadInput{AllocatedContext: 1000, ReadContext: 1000, TokensPerIteration: 1, MinPerSessionTokps: 120},
	}
	result, err := calculateDecodeUpperBound(input)
	if err != nil {
		t.Fatalf("calculateDecodeUpperBound: %v", err)
	}
	if result.State != "usable" || result.ResidentWeightGB != 7 || result.MemoryFitBatch != 15 || result.UsableBatch != 13 {
		t.Fatalf("unexpected dense result: %+v", result)
	}
	wantSingle := 1000.0 / 7.100004
	if result.SingleSession == nil || math.Abs(result.SingleSession.PerSessionTokps-wantSingle) >= 1e-9 {
		t.Fatalf("single-session speed = %#v, want %.12f", result.SingleSession, wantSingle)
	}
	if result.Optimum == nil || result.Optimum.PerSessionTokps < 120 {
		t.Fatalf("optimum = %#v, want speed floor satisfied", result.Optimum)
	}
}

func TestDecodeCalculatorSpeculativeAndFailureGuidance(t *testing.T) {
	if got := expectedSpeculativeTokensCLI(4, 0.8); math.Abs(got-3.3616) >= 1e-12 {
		t.Fatalf("expected speculative tokens = %.12f, want 3.3616", got)
	}

	cases := []struct {
		name       string
		capacity   string
		minimum    string
		wantState  string
		constraint string
	}{
		{name: "resident", capacity: "10", wantState: "resident-fail", constraint: "memory"},
		{name: "session", capacity: "25", wantState: "session-fail", constraint: "kv-cache"},
		{name: "speed", capacity: "128", minimum: "1000", wantState: "floor-fail", constraint: "speed"},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			argv := []string{"calculate", "decode", "--capacity-gb", tc.capacity}
			if tc.minimum != "" {
				argv = append(argv, "--min-tok-s", tc.minimum)
			}
			input, err := decodeInputFromArgs(parseArgs(argv))
			if err != nil {
				t.Fatalf("decodeInputFromArgs: %v", err)
			}
			result, err := calculateDecodeUpperBound(input)
			if err != nil {
				t.Fatalf("calculateDecodeUpperBound: %v", err)
			}
			if result.State != tc.wantState {
				t.Fatalf("state = %q, want %q", result.State, tc.wantState)
			}
			guidance := decodeResultConstraint(input, result)
			if guidance == nil || guidance.Kind != tc.constraint || len(guidance.Suggestions) == 0 {
				t.Fatalf("guidance = %#v, want kind %q with suggestions", guidance, tc.constraint)
			}
		})
	}
}

func TestDecodeCalculatorAgentContract(t *testing.T) {
	args := parseArgs([]string{
		"calculate", "decode", "--decoding", "speculative", "--draft-tokens", "4",
		"--acceptance-percent", "80", "--draft-resident-gb", "1", "--draft-traffic-gb", "0.5", "--json",
	})
	if err := validateOptionNames(args); err != nil {
		t.Fatalf("agent flags rejected: %v", err)
	}
	input, err := decodeInputFromArgs(args)
	if err != nil {
		t.Fatalf("decodeInputFromArgs: %v", err)
	}
	if math.Abs(input.Workload.TokensPerIteration-3.3616) >= 1e-12 {
		t.Fatalf("tokensPerIteration = %.12f", input.Workload.TokensPerIteration)
	}
	if !knownTopLevel("calculate") || commandDescriptions["calculate decode"] == "" {
		t.Fatal("calculate decode is missing from command discovery")
	}
}

func TestDecodeCalculatorRejectsInconsistentContext(t *testing.T) {
	_, err := decodeInputFromArgs(parseArgs([]string{"calculate", "decode", "--allocated-context", "1024", "--read-context", "2048"}))
	if err == nil {
		t.Fatal("expected inconsistent context error")
	}
	cliErr, ok := err.(cliError)
	if !ok || cliErr.Code != "invalid_decode_input" {
		t.Fatalf("error = %T %v, want invalid_decode_input", err, err)
	}
}
