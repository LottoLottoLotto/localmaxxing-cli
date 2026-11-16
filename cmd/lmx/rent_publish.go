package main

import (
	"bytes"
	"context"
	"encoding/json"
	"fmt"
	"io"
	"math"
	"net/http"
	"os"
	"sort"
	"strings"
	"time"
)

// ── rent publish ────────────────────────────────────────────────────────────

// rentPublishedModel is one model of a listing as the site stores it: credits per 1,000 tokens.
type rentPublishedModel struct {
	Model                       string   `json:"model"`
	PricePer1kInputTokens       float64  `json:"pricePer1kInputTokens"`
	PricePer1kCachedInputTokens *float64 `json:"pricePer1kCachedInputTokens"`
	PricePer1kOutputTokens      float64  `json:"pricePer1kOutputTokens"`
	ContextLength               *int     `json:"contextLength"`
}

// emxPricedModel is a model in an emx gateway's /v1/models, with the price an admin key sees (emx price).
type emxPricedModel struct {
	ID          string `json:"id"`
	MaxModelLen int    `json:"max_model_len"`
	EMX         struct {
		Price *struct {
			Prompt     float64  `json:"prompt"`
			Completion float64  `json:"completion"`
			Cached     *float64 `json:"cached"`
			Currency   string   `json:"currency"`
		} `json:"price"`
	} `json:"emx"`
}

// creditsPer1k turns an emx price (per million tokens, in USD) into the site's unit: credits per 1,000
// tokens, 1 credit = $0.01. $3 per million = $0.003 per thousand = 0.3 credits per thousand.
func creditsPer1k(usdPerMillion float64) float64 {
	return math.Round(usdPerMillion/10*1e6) / 1e6
}

// handleRentPublish sets a listing's models and their prices from the emx gateway: the models running
// there (or --model), each at its `emx price`, converted to credits per 1k tokens, with its context
// window. The site then bills each renter request at the price of the model it names and refuses the rest.
func handleRentPublish(args cliArgs) error {
	listing := positional(args, 2)
	if listing == "" {
		return cliError{"missing_listing", "lmx rent publish needs a listing ID.", []string{"Find your listing IDs with lmx rent list --mine."}, nil}
	}
	key := apiKey(args)
	if key == "" {
		return missingAPIKey("lmx rent publish updates your listing with your LocalMaxxing API key.")
	}
	cfg := rentHostConfig{Target: openAIBaseURL(firstNonEmpty(opt(args, "target"), gatewayURL(args))),
		TargetKey: firstNonEmpty(opt(args, "target-key"), os.Getenv("LMX_RENT_TARGET_KEY")), DiscoveryKey: os.Getenv("EMX_API_KEY"),
		Client: &http.Client{}}
	if cfg.DiscoveryKey == "" {
		return cliError{"missing_emx_key", "lmx rent publish reads model prices from emx, which shows them to admin keys only.", []string{"export EMX_API_KEY=<emx admin or auditor key> and retry."}, nil}
	}
	models := optValues(args, "model")
	if len(models) == 0 {
		var err error
		if models, err = rentTargetModels(cfg); err != nil {
			return err
		}
	}
	priced, err := rentEMXPrices(cfg)
	if err != nil {
		return err
	}
	out := []rentPublishedModel{}
	unpriced := []string{}
	for _, name := range models {
		m, ok := priced[name]
		if !ok {
			return cliError{"model_not_on_target", "Model " + name + " is not on the emx gateway at " + cfg.Target + ".", []string{"See emx models, or pass the names it serves with --model."}, nil}
		}
		if m.EMX.Price == nil {
			unpriced = append(unpriced, name)
			continue
		}
		if cur := strings.ToUpper(m.EMX.Price.Currency); cur != "USD" {
			return cliError{"price_currency", "emx prices are in " + m.EMX.Price.Currency + "; rentals are paid in credits worth $0.01, so set them in USD.", []string{"emx price set " + name + " --prompt <USD per 1M> --completion <USD per 1M> --currency USD"}, nil}
		}
		pm := rentPublishedModel{Model: name, PricePer1kInputTokens: creditsPer1k(m.EMX.Price.Prompt), PricePer1kOutputTokens: creditsPer1k(m.EMX.Price.Completion)}
		if m.EMX.Price.Cached != nil {
			cached := creditsPer1k(*m.EMX.Price.Cached)
			pm.PricePer1kCachedInputTokens = &cached
		}
		if m.MaxModelLen > 0 {
			ctx := m.MaxModelLen
			pm.ContextLength = &ctx
		}
		out = append(out, pm)
	}
	if len(unpriced) > 0 {
		hints := []string{}
		for _, name := range unpriced {
			hints = append(hints, "emx price set "+name+" --prompt <USD per 1M tokens> --completion <USD per 1M tokens>")
		}
		return cliError{"model_unpriced", "No emx price for " + strings.Join(unpriced, ", ") + ": a listing never offers a model at a price nobody set.", append(hints, "Or leave it out: --model <the others>."), nil}
	}
	sort.Slice(out, func(i, j int) bool { return out[i].Model < out[j].Model })

	result := map[string]any{"listing": listing, "models": out, "applied": false}
	if !hasFlag(args, "dry-run") {
		if err := rentPatchListingModels(cfg.Client, apiURL(args), key, listing, out); err != nil {
			return err
		}
		result["applied"] = true
	}
	if hasFlag(args, "json") {
		printJSON(args, result)
		return nil
	}
	verb := "Listing " + listing + " now offers"
	if hasFlag(args, "dry-run") {
		verb = "Would set listing " + listing + " to offer"
	}
	fmt.Printf("%s (credits per 1k tokens; 1 credit = $0.01):\n", verb)
	fmt.Printf("  %-32s %10s %10s %10s %9s\n", "MODEL", "INPUT", "CACHED", "OUTPUT", "CONTEXT")
	for _, m := range out {
		cached, ctx := "= input", "-"
		if m.PricePer1kCachedInputTokens != nil {
			cached = fmt.Sprint(*m.PricePer1kCachedInputTokens)
		}
		if m.ContextLength != nil {
			ctx = fmt.Sprint(*m.ContextLength)
		}
		fmt.Printf("  %-32s %10v %10s %10v %9s\n", m.Model, m.PricePer1kInputTokens, cached, m.PricePer1kOutputTokens, ctx)
	}
	return nil
}

// rentEMXPrices reads the gateway's models with their prices and context windows (admin view).
func rentEMXPrices(cfg rentHostConfig) (map[string]emxPricedModel, error) {
	res, cancel, err := rentTargetGet(cfg, "/v1/models?unavailable=1")
	if err != nil {
		return nil, cliError{"target_unreachable", "Could not reach the emx gateway at " + cfg.Target + ": " + err.Error(), []string{"Start it (emx serve) or pass --target <base-url>."}, nil}
	}
	defer cancel()
	defer res.Body.Close()
	var list struct {
		Data []emxPricedModel `json:"data"`
	}
	if res.StatusCode != http.StatusOK || json.NewDecoder(io.LimitReader(res.Body, 8<<20)).Decode(&list) != nil {
		return nil, cliError{"target_models_failed", fmt.Sprintf("%s/v1/models returned HTTP %d.", cfg.Target, res.StatusCode), []string{"EMX_API_KEY must be an emx admin or auditor key."}, nil}
	}
	out := map[string]emxPricedModel{}
	for _, m := range list.Data {
		out[m.ID] = m
	}
	return out, nil
}

// rentPatchListingModels replaces the listing's models and prices on the site.
func rentPatchListingModels(client *http.Client, siteURL, key, listing string, models []rentPublishedModel) error {
	body, _ := json.Marshal(map[string]any{"models": models})
	ctx, cancel := context.WithTimeout(context.Background(), 30*time.Second)
	defer cancel()
	req, err := http.NewRequestWithContext(ctx, http.MethodPatch, strings.TrimRight(siteURL, "/")+"/api/rentals/"+listing, bytes.NewReader(body))
	if err != nil {
		return err
	}
	req.Header.Set("Authorization", "Bearer "+key)
	req.Header.Set("Content-Type", "application/json")
	res, err := client.Do(req)
	if err != nil {
		return cliError{"site_unreachable", "Could not reach " + siteURL + ": " + err.Error(), nil, nil}
	}
	defer res.Body.Close()
	answer, _ := io.ReadAll(io.LimitReader(res.Body, 1<<20))
	if res.StatusCode != http.StatusOK {
		return cliError{"publish_failed", fmt.Sprintf("The site refused the update (HTTP %d): %s", res.StatusCode, strings.TrimSpace(string(answer))), []string{"The site allows one listing update per minute."}, nil}
	}
	return nil
}
