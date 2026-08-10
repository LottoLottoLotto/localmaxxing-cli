package main

import (
	"fmt"
	"math"
	"strconv"
	"strings"
)

const decimalGB = 1_000_000_000

type decodeCalculatorInput struct {
	Hardware decodeHardwareInput `json:"hardware"`
	Model    decodeModelInput    `json:"model"`
	Cache    decodeCacheInput    `json:"cache"`
	Workload decodeWorkloadInput `json:"workload"`
}

type decodeHardwareInput struct {
	CapacityGB    float64 `json:"capacityGb"`
	BandwidthGBps float64 `json:"bandwidthGbps"`
	OverheadGB    float64 `json:"overheadGb"`
}

type decodeModelInput struct {
	Architecture    string  `json:"architecture"`
	TotalParamsB    float64 `json:"totalParamsB"`
	ActiveParamsB   float64 `json:"activeParamsB,omitempty"`
	WeightBits      float64 `json:"weightBits"`
	DraftResidentGB float64 `json:"draftResidentGb"`
	DraftTrafficGB  float64 `json:"draftTrafficGb"`
	ExpertCount     int     `json:"expertCount,omitempty"`
	ExpertsPerToken int     `json:"expertsPerToken,omitempty"`
	MoERouting      string  `json:"moeRouting"`
}

type decodeCacheInput struct {
	GlobalLayers      int     `json:"globalLayers"`
	LocalLayers       int     `json:"localLayers"`
	KVHeads           int     `json:"kvHeads"`
	HeadDim           int     `json:"headDim"`
	StoreBits         float64 `json:"storeBits"`
	ReadBits          float64 `json:"readBits"`
	SlidingWindow     int     `json:"slidingWindow"`
	FixedStateStoreGB float64 `json:"fixedStateStoreGb"`
	FixedStateReadGB  float64 `json:"fixedStateReadGb"`
}

type decodeWorkloadInput struct {
	AllocatedContext   int     `json:"allocatedContext"`
	ReadContext        int     `json:"readContext"`
	TokensPerIteration float64 `json:"tokensPerIteration"`
	MinPerSessionTokps float64 `json:"minPerSessionTokps"`
}

type decodePoint struct {
	Batch                 int     `json:"batch"`
	WeightTrafficGB       float64 `json:"weightTrafficGb"`
	KVReadGB              float64 `json:"kvReadGb"`
	BytesPerOutputTokenGB float64 `json:"bytesPerOutputTokenGb"`
	AggregateTokps        float64 `json:"aggregateTokps"`
	PerSessionTokps       float64 `json:"perSessionTokps"`
}

type decodeCalculatorResult struct {
	State                  string       `json:"state"`
	ResidentWeightGB       float64      `json:"residentWeightGb"`
	ActiveWeightGB         float64      `json:"activeWeightGb"`
	ResidentMarginGB       float64      `json:"residentMarginGb"`
	KVAllocPerSessionGB    float64      `json:"kvAllocPerSessionGb"`
	KVReadPerTokenGB       float64      `json:"kvReadPerTokenGb"`
	MemoryFitBatch         int          `json:"memoryFitBatch"`
	RateLimitedBatch       int          `json:"rateLimitedBatch"`
	UsableBatch            int          `json:"usableBatch"`
	SingleSession          *decodePoint `json:"singleSession"`
	Optimum                *decodePoint `json:"optimum"`
	MemoryPowerGB2PerSec   float64      `json:"memoryPowerGb2PerSec"`
	MemoryPowerBoundTokps  *float64     `json:"memoryPowerBoundTokps"`
	LargeMemoryBoundTokps  *float64     `json:"largeMemoryBoundTokps"`
	UsesExpectedMoERouting bool         `json:"usesExpectedMoeRouting"`
}

type decodeConstraint struct {
	Kind        string         `json:"kind"`
	Title       string         `json:"title"`
	Description string         `json:"description"`
	Metrics     map[string]any `json:"metrics"`
	Suggestions []string       `json:"suggestions"`
}

type decodeCalculationPayload struct {
	SchemaVersion int                    `json:"schemaVersion"`
	Kind          string                 `json:"kind"`
	Input         decodeCalculatorInput  `json:"input"`
	Result        decodeCalculatorResult `json:"result"`
	Constraint    *decodeConstraint      `json:"constraint"`
}

func defaultDecodeCalculatorInput() decodeCalculatorInput {
	return decodeCalculatorInput{
		Hardware: decodeHardwareInput{CapacityGB: 128, BandwidthGBps: 800, OverheadGB: 6},
		Model: decodeModelInput{
			Architecture: "dense", TotalParamsB: 32, ActiveParamsB: 32, WeightBits: 4.5,
			MoERouting: "expected", ExpertCount: 256, ExpertsPerToken: 8,
		},
		Cache: decodeCacheInput{
			GlobalLayers: 64, KVHeads: 8, HeadDim: 128, StoreBits: 16, ReadBits: 16,
			SlidingWindow: 4096,
		},
		Workload: decodeWorkloadInput{
			AllocatedContext: 32768, ReadContext: 16000, TokensPerIteration: 1, MinPerSessionTokps: 20,
		},
	}
}

func handleDecodeCalculator(args cliArgs) error {
	input, err := decodeInputFromArgs(args)
	if err != nil {
		return err
	}
	result, err := calculateDecodeUpperBound(input)
	if err != nil {
		return err
	}
	payload := decodeCalculationPayload{
		SchemaVersion: 1,
		Kind:          "decode_upper_bound",
		Input:         input,
		Result:        result,
		Constraint:    decodeResultConstraint(input, result),
	}
	if hasFlag(args, "json") || hasFlag(args, "compact") || opt(args, "out") != "" {
		return writeOrPrintJSON("decode_calculation", args, payload)
	}
	printDecodeCalculation(payload)
	return nil
}

func decodeInputFromArgs(args cliArgs) (decodeCalculatorInput, error) {
	input := defaultDecodeCalculatorInput()
	var err error
	if input.Hardware.CapacityGB, err = decodeFloatOption(args, "capacity-gb", input.Hardware.CapacityGB, 0, false); err != nil {
		return input, err
	}
	if input.Hardware.BandwidthGBps, err = decodeFloatOption(args, "bandwidth-gbps", input.Hardware.BandwidthGBps, 0, false); err != nil {
		return input, err
	}
	if input.Hardware.OverheadGB, err = decodeFloatOption(args, "overhead-gb", input.Hardware.OverheadGB, 0, true); err != nil {
		return input, err
	}
	if input.Model.TotalParamsB, err = decodeFloatOption(args, "total-params-b", input.Model.TotalParamsB, 0, false); err != nil {
		return input, err
	}
	if input.Model.ActiveParamsB, err = decodeFloatOption(args, "active-params-b", input.Model.ActiveParamsB, 0, false); err != nil {
		return input, err
	}
	if input.Model.WeightBits, err = decodeFloatOption(args, "weight-bits", input.Model.WeightBits, 0, false); err != nil {
		return input, err
	}
	if input.Model.DraftResidentGB, err = decodeFloatOption(args, "draft-resident-gb", input.Model.DraftResidentGB, 0, true); err != nil {
		return input, err
	}
	if input.Model.DraftTrafficGB, err = decodeFloatOption(args, "draft-traffic-gb", input.Model.DraftTrafficGB, 0, true); err != nil {
		return input, err
	}
	if input.Cache.StoreBits, err = decodeFloatOption(args, "kv-store-bits", input.Cache.StoreBits, 0, false); err != nil {
		return input, err
	}
	if input.Cache.ReadBits, err = decodeFloatOption(args, "kv-read-bits", input.Cache.ReadBits, 0, false); err != nil {
		return input, err
	}
	if input.Cache.FixedStateStoreGB, err = decodeFloatOption(args, "fixed-state-store-gb", input.Cache.FixedStateStoreGB, 0, true); err != nil {
		return input, err
	}
	if input.Cache.FixedStateReadGB, err = decodeFloatOption(args, "fixed-state-read-gb", input.Cache.FixedStateReadGB, 0, true); err != nil {
		return input, err
	}
	if input.Workload.MinPerSessionTokps, err = decodeFloatOption(args, "min-tok-s", input.Workload.MinPerSessionTokps, 0, false); err != nil {
		return input, err
	}

	if input.Cache.GlobalLayers, err = intOption(args, input.Cache.GlobalLayers, 0, "global-layers"); err != nil {
		return input, err
	}
	if input.Cache.LocalLayers, err = intOption(args, input.Cache.LocalLayers, 0, "local-layers"); err != nil {
		return input, err
	}
	if input.Cache.KVHeads, err = intOption(args, input.Cache.KVHeads, 1, "kv-heads"); err != nil {
		return input, err
	}
	if input.Cache.HeadDim, err = intOption(args, input.Cache.HeadDim, 1, "head-dim"); err != nil {
		return input, err
	}
	if input.Cache.SlidingWindow, err = intOption(args, input.Cache.SlidingWindow, 1, "sliding-window"); err != nil {
		return input, err
	}
	if input.Workload.AllocatedContext, err = intOption(args, input.Workload.AllocatedContext, 1, "allocated-context"); err != nil {
		return input, err
	}
	if input.Workload.ReadContext, err = intOption(args, input.Workload.ReadContext, 1, "read-context"); err != nil {
		return input, err
	}
	if input.Model.ExpertCount, err = intOption(args, input.Model.ExpertCount, 1, "expert-count"); err != nil {
		return input, err
	}
	if input.Model.ExpertsPerToken, err = intOption(args, input.Model.ExpertsPerToken, 1, "experts-per-token"); err != nil {
		return input, err
	}

	if value := strings.ToLower(opt(args, "architecture")); value != "" {
		if value != "dense" && value != "moe" {
			return input, cliError{"invalid_option", "--architecture must be dense or moe", []string{"Pass --architecture dense or --architecture moe."}, value}
		}
		input.Model.Architecture = value
	}
	if value := strings.ToLower(opt(args, "moe-routing")); value != "" {
		if value != "expected" && value != "minimum" {
			return input, cliError{"invalid_option", "--moe-routing must be expected or minimum", nil, value}
		}
		input.Model.MoERouting = value
	}
	decoding := strings.ToLower(opt(args, "decoding"))
	if decoding == "" || decoding == "standard" {
		input.Workload.TokensPerIteration = 1
	} else if decoding == "speculative" {
		draftTokens, parseErr := intOption(args, 4, 1, "draft-tokens")
		if parseErr != nil {
			return input, parseErr
		}
		acceptance, parseErr := decodeFloatOption(args, "acceptance-percent", 80, 0, true)
		if parseErr != nil {
			return input, parseErr
		}
		if acceptance > 100 {
			return input, cliError{"invalid_option", "--acceptance-percent cannot exceed 100", nil, acceptance}
		}
		input.Workload.TokensPerIteration = expectedSpeculativeTokensCLI(draftTokens, acceptance/100)
	} else {
		return input, cliError{"invalid_option", "--decoding must be standard or speculative", nil, decoding}
	}
	return input, validateDecodeInput(input)
}

func decodeFloatOption(args cliArgs, key string, fallback, minimum float64, allowEqual bool) (float64, error) {
	value := opt(args, key)
	if value == "" {
		return fallback, nil
	}
	parsed, err := strconv.ParseFloat(value, 64)
	valid := err == nil && !math.IsNaN(parsed) && !math.IsInf(parsed, 0)
	if allowEqual {
		valid = valid && parsed >= minimum
	} else {
		valid = valid && parsed > minimum
	}
	if !valid {
		relation := "greater than"
		if allowEqual {
			relation = "at least"
		}
		return 0, cliError{"invalid_option", fmt.Sprintf("--%s must be %s %g", key, relation, minimum), nil, value}
	}
	return parsed, nil
}

func expectedSpeculativeTokensCLI(draftTokens int, acceptance float64) float64 {
	if acceptance == 1 {
		return float64(draftTokens + 1)
	}
	return (1 - math.Pow(acceptance, float64(draftTokens+1))) / (1 - acceptance)
}

func validateDecodeInput(input decodeCalculatorInput) error {
	if input.Cache.GlobalLayers+input.Cache.LocalLayers < 1 {
		return cliError{"invalid_decode_input", "At least one attention layer is required", nil, nil}
	}
	if input.Workload.ReadContext > input.Workload.AllocatedContext {
		return cliError{"invalid_decode_input", "Read context cannot exceed allocated context", []string{"Lower --read-context or raise --allocated-context."}, nil}
	}
	if input.Model.Architecture == "moe" {
		if input.Model.ActiveParamsB > input.Model.TotalParamsB {
			return cliError{"invalid_decode_input", "Active parameters cannot exceed total parameters", nil, nil}
		}
		if input.Model.ExpertsPerToken >= input.Model.ExpertCount {
			return cliError{"invalid_decode_input", "Experts per token must be smaller than expert count", nil, nil}
		}
		expertParamsB := (input.Model.TotalParamsB - input.Model.ActiveParamsB) / float64(input.Model.ExpertCount-input.Model.ExpertsPerToken)
		if input.Model.ActiveParamsB-float64(input.Model.ExpertsPerToken)*expertParamsB < 0 {
			return cliError{"invalid_decode_input", "MoE fields imply a negative fixed-parameter trunk", nil, nil}
		}
	}
	return nil
}

func calculateDecodeUpperBound(input decodeCalculatorInput) (decodeCalculatorResult, error) {
	if err := validateDecodeInput(input); err != nil {
		return decodeCalculatorResult{}, err
	}
	bytesPerWeight := input.Model.WeightBits / 8
	residentWeightGB := input.Model.TotalParamsB*bytesPerWeight + input.Model.DraftResidentGB
	activeWeightGB := input.Model.TotalParamsB*bytesPerWeight + input.Model.DraftTrafficGB
	weightTraffic := func(batch int) float64 { return activeWeightGB }
	if input.Model.Architecture == "moe" {
		active := input.Model.ActiveParamsB
		experts := input.Model.ExpertCount
		selected := input.Model.ExpertsPerToken
		expertParamsB := (input.Model.TotalParamsB - active) / float64(experts-selected)
		fixedParamsB := active - float64(selected)*expertParamsB
		activeWeightGB = active*bytesPerWeight + input.Model.DraftTrafficGB
		weightTraffic = func(batch int) float64 {
			if input.Model.MoERouting == "minimum" {
				return activeWeightGB
			}
			routings := float64(batch) * input.Workload.TokensPerIteration
			distinctExperts := float64(experts) * (1 - math.Pow(1-float64(selected)/float64(experts), routings))
			return (fixedParamsB+expertParamsB*distinctExperts)*bytesPerWeight + input.Model.DraftTrafficGB
		}
	}
	kvFootprint := func(context int, bits, fixedGB float64) float64 {
		attended := input.Cache.GlobalLayers*context + input.Cache.LocalLayers*min(context, input.Cache.SlidingWindow)
		bytes := 2 * float64(input.Cache.KVHeads) * float64(input.Cache.HeadDim) * (bits / 8) * float64(attended)
		return bytes/decimalGB + fixedGB
	}
	residentMarginGB := input.Hardware.CapacityGB - residentWeightGB - input.Hardware.OverheadGB
	kvAllocGB := kvFootprint(input.Workload.AllocatedContext, input.Cache.StoreBits, input.Cache.FixedStateStoreGB)
	kvReadGB := kvFootprint(input.Workload.ReadContext, input.Cache.ReadBits, input.Cache.FixedStateReadGB)
	memoryPower := input.Hardware.CapacityGB * input.Hardware.BandwidthGBps
	result := decodeCalculatorResult{
		ResidentWeightGB: residentWeightGB, ActiveWeightGB: activeWeightGB, ResidentMarginGB: residentMarginGB,
		KVAllocPerSessionGB: kvAllocGB, KVReadPerTokenGB: kvReadGB, MemoryPowerGB2PerSec: memoryPower,
		UsesExpectedMoERouting: input.Model.Architecture == "moe" && input.Model.MoERouting == "expected",
	}
	if residentMarginGB < 0 {
		result.State = "resident-fail"
		return result, nil
	}
	memoryFitBatch := max(0, int(math.Floor(residentMarginGB/kvAllocGB)))
	memoryPowerBound := input.Workload.TokensPerIteration * input.Hardware.BandwidthGBps * residentMarginGB / (kvAllocGB * activeWeightGB)
	largeMemoryBound := input.Workload.TokensPerIteration * memoryPower / (kvAllocGB * activeWeightGB)
	result.MemoryFitBatch = memoryFitBatch
	result.MemoryPowerBoundTokps = &memoryPowerBound
	result.LargeMemoryBoundTokps = &largeMemoryBound
	if memoryFitBatch < 1 {
		result.State = "session-fail"
		return result, nil
	}
	point := func(batch int) decodePoint {
		traffic := weightTraffic(batch)
		bytesPerToken := traffic/(float64(batch)*input.Workload.TokensPerIteration) + kvReadGB
		aggregate := input.Hardware.BandwidthGBps / bytesPerToken
		return decodePoint{Batch: batch, WeightTrafficGB: traffic, KVReadGB: kvReadGB, BytesPerOutputTokenGB: bytesPerToken, AggregateTokps: aggregate, PerSessionTokps: aggregate / float64(batch)}
	}
	single := point(1)
	result.SingleSession = &single
	closedRateBatch := math.MaxInt
	if kvReadGB > 0 {
		closedRateBatch = int(math.Floor((input.Workload.TokensPerIteration*input.Hardware.BandwidthGBps/input.Workload.MinPerSessionTokps - activeWeightGB) / (input.Workload.TokensPerIteration * kvReadGB)))
	}
	result.RateLimitedBatch = max(0, closedRateBatch)
	if single.PerSessionTokps < input.Workload.MinPerSessionTokps {
		result.State = "floor-fail"
		return result, nil
	}
	low, high := 1, memoryFitBatch
	for low < high {
		mid := low + int(math.Ceil(float64(high-low)/2))
		if point(mid).PerSessionTokps >= input.Workload.MinPerSessionTokps {
			low = mid
		} else {
			high = mid - 1
		}
	}
	optimum := point(low)
	result.State = "usable"
	result.UsableBatch = low
	result.Optimum = &optimum
	return result, nil
}

func decodeResultConstraint(input decodeCalculatorInput, result decodeCalculatorResult) *decodeConstraint {
	switch result.State {
	case "resident-fail":
		required := result.ResidentWeightGB + input.Hardware.OverheadGB
		return &decodeConstraint{Kind: "memory", Title: "The model cannot be loaded", Description: fmt.Sprintf("Weights and runtime need %.2f GB, but the hardware has %.2f GB.", required, input.Hardware.CapacityGB), Metrics: map[string]any{"availableGb": input.Hardware.CapacityGB, "requiredGb": required, "shortfallGb": math.Abs(result.ResidentMarginGB)}, Suggestions: []string{"Lower --weight-bits or --total-params-b.", "Reduce --overhead-gb if the reserve is conservative.", fmt.Sprintf("Use hardware with at least %.2f GB capacity.", required)}}
	case "session-fail":
		remaining := math.Max(0, result.ResidentMarginGB)
		return &decodeConstraint{Kind: "kv-cache", Title: "The model fits, but no session does", Description: fmt.Sprintf("%.2f GB remains after weights; one KV reservation needs %.2f GB.", remaining, result.KVAllocPerSessionGB), Metrics: map[string]any{"remainingGb": remaining, "sessionGb": result.KVAllocPerSessionGB, "shortfallGb": math.Max(0, result.KVAllocPerSessionGB-result.ResidentMarginGB)}, Suggestions: []string{"Reduce --allocated-context.", "Lower --kv-store-bits or reduce attention layers.", "Use hardware with more memory."}}
	case "floor-fail":
		best := 0.0
		if result.SingleSession != nil {
			best = result.SingleSession.PerSessionTokps
		}
		return &decodeConstraint{Kind: "speed", Title: "The requested speed is out of reach", Description: fmt.Sprintf("One session reaches %.2f tok/s, below the %.2f tok/s minimum.", best, input.Workload.MinPerSessionTokps), Metrics: map[string]any{"bestTokps": best, "minimumTokps": input.Workload.MinPerSessionTokps, "gapTokps": math.Max(0, input.Workload.MinPerSessionTokps-best)}, Suggestions: []string{"Lower --min-tok-s.", "Increase --bandwidth-gbps or reduce active model weights.", "Reduce --read-context to lower KV traffic."}}
	default:
		return nil
	}
}

func printDecodeCalculation(payload decodeCalculationPayload) {
	result := payload.Result
	fmt.Println("Decode upper-bound calculator")
	fmt.Printf("State: %s\n", result.State)
	if result.Optimum != nil {
		fmt.Printf("Useful throughput: %.2f tok/s aggregate\n", result.Optimum.AggregateTokps)
		fmt.Printf("Useful batch: %d sessions at %.2f tok/s each\n", result.Optimum.Batch, result.Optimum.PerSessionTokps)
	} else if payload.Constraint != nil {
		fmt.Printf("Blocked: %s\n", payload.Constraint.Title)
		fmt.Println(payload.Constraint.Description)
		fmt.Println("What to change:")
		for _, suggestion := range payload.Constraint.Suggestions {
			fmt.Println("  - " + suggestion)
		}
	}
	fmt.Printf("Resident weights: %.2f GB; KV per session: %.2f GB\n", result.ResidentWeightGB, result.KVAllocPerSessionGB)
	fmt.Println("Use --json for the complete machine-readable calculation.")
}
