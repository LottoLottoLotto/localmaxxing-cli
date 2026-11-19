package main

import (
	"bytes"
	"context"
	"encoding/json"
	"fmt"
	"io"
	"net/http"
	"os"
	"sort"
	"strconv"
	"strings"
	"time"
)

// ── rent publish ────────────────────────────────────────────────────────────

// rentPublishedModel is one model of a listing as the site stores it: US dollars per 1M tokens, and the
// weights the host serves it from (emx's report; the site verifies them against Hugging Face). Weights are
// always sent, null when emx has none ready, so the site never keeps weights from an earlier publish.
type rentPublishedModel struct {
	Model                       string           `json:"model"`
	PricePer1MInputTokens       float64          `json:"pricePer1MInputTokens"`
	PricePer1MCachedInputTokens *float64         `json:"pricePer1MCachedInputTokens"`
	PricePer1MOutputTokens      float64          `json:"pricePer1MOutputTokens"`
	ContextLength               *int             `json:"contextLength"`
	Weights                     json.RawMessage  `json:"weights"`
	Source                      *rentModelSource `json:"source,omitempty"`
}

// rentModelSource is the Hugging Face repo (and commit) a model's weights claim to come from.
type rentModelSource struct {
	Repo     string `json:"repo"`
	Revision string `json:"revision,omitempty"`
}

// emxWeights is the part of emx's `weights` report lmx reads; the whole object is published as reported.
type emxWeights struct {
	Status string           `json:"status"`
	ID     string           `json:"id"`
	Origin *rentModelSource `json:"origin"`
	Error  string           `json:"error"`
}

// emxPricedModel is a model in an emx gateway's /v1/models, with the price an admin key sees (emx price)
// and the identity of its weight files (emx hashes them; absent on gateways that do not).
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
		Weights json.RawMessage `json:"weights"`
	} `json:"emx"`
}

// weights parses the model's emx weights report (nil when emx reports none).
func (m emxPricedModel) weights() *emxWeights {
	if len(m.EMX.Weights) == 0 || string(m.EMX.Weights) == "null" {
		return nil
	}
	var w emxWeights
	if json.Unmarshal(m.EMX.Weights, &w) != nil {
		return nil
	}
	return &w
}

// weightsStatus is the model's weights status for display: ready, hashing, error, or "-" (not reported).
func (m emxPricedModel) weightsStatus() string {
	if w := m.weights(); w != nil && w.Status != "" {
		return w.Status
	}
	return "-"
}

// rentHashingPoll is how often rent publish re-reads emx while it waits for weights to finish hashing.
var rentHashingPoll = 2 * time.Second

// rentParseSources reads --source <model>=<repo>[@<revision>] (repeatable).
func rentParseSources(args cliArgs) (map[string]rentModelSource, error) {
	out := map[string]rentModelSource{}
	for _, value := range optValues(args, "source") {
		model, ref, ok := strings.Cut(value, "=")
		repo, revision, _ := strings.Cut(strings.TrimSpace(ref), "@")
		model, repo, revision = strings.TrimSpace(model), strings.TrimSpace(repo), strings.TrimSpace(revision)
		owner, name, slash := strings.Cut(repo, "/")
		if !ok || model == "" || !slash || owner == "" || name == "" || strings.Contains(name, "/") || (strings.Contains(ref, "@") && revision == "") {
			return nil, cliError{"invalid_option", "--source must be <model>=<owner>/<repo>[@<revision>], e.g. qwen3-0.6b=unsloth/Qwen3-0.6B-GGUF.", nil, value}
		}
		out[model] = rentModelSource{Repo: repo, Revision: revision}
	}
	return out, nil
}

// rentEMXPricesSettled reads the gateway's models, waiting up to wait while any of models still has its
// weights hashing.
func rentEMXPricesSettled(cfg rentHostConfig, models []string, wait time.Duration) (map[string]emxPricedModel, error) {
	deadline := time.Now().Add(wait)
	for {
		priced, err := rentEMXPrices(cfg)
		if err != nil {
			return nil, err
		}
		hashing := false
		for _, name := range models {
			if m, ok := priced[name]; ok && m.weightsStatus() == "hashing" {
				hashing = true
			}
		}
		if !hashing || !time.Now().Before(deadline) {
			return priced, nil
		}
		time.Sleep(min(rentHashingPoll, time.Until(deadline)))
	}
}

// handleRentPublish sets a listing's models and their prices from the emx gateway: the models running
// there (or --model), each at its `emx price` (US dollars per 1M tokens, the site's unit too), with its context
// window and the weights emx hashed (the site checks them against Hugging Face: Verified, else Custom).
// The site then bills each renter request at the price of the model it names and refuses the rest.
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
	sources, err := rentParseSources(args)
	if err != nil {
		return err
	}
	for model := range sources {
		if !sliceContainsString(models, model) {
			return cliError{"invalid_option", "--source names " + model + ", which this listing would not offer.", []string{"Publishing: " + strings.Join(models, ", ") + "."}, nil}
		}
	}
	wait := 120 * time.Second
	if value := opt(args, "wait-hashing"); value == "0" {
		wait = 0
	} else if wait, err = authDurationOption(args, "wait-hashing", wait); err != nil {
		return err
	}
	priced, err := rentEMXPricesSettled(cfg, models, wait)
	if err != nil {
		return err
	}
	out := []rentPublishedModel{}
	unpriced := []string{}
	weightsStatus := map[string]string{}
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
			return cliError{"price_currency", "emx prices are in " + m.EMX.Price.Currency + "; rentals are priced in US dollars, so set them in USD.", []string{"emx price set " + name + " --prompt <USD per 1M> --completion <USD per 1M> --currency USD"}, nil}
		}
		pm := rentPublishedModel{Model: name, PricePer1MInputTokens: m.EMX.Price.Prompt, PricePer1MOutputTokens: m.EMX.Price.Completion}
		if m.EMX.Price.Cached != nil {
			cached := *m.EMX.Price.Cached
			pm.PricePer1MCachedInputTokens = &cached
		}
		if m.MaxModelLen > 0 {
			ctx := m.MaxModelLen
			pm.ContextLength = &ctx
		}
		weights := m.weights()
		if weights != nil && weights.Status == "ready" && weights.ID != "" {
			pm.Weights = m.EMX.Weights
			if weights.Origin != nil && weights.Origin.Repo != "" {
				origin := *weights.Origin
				pm.Source = &origin
			}
		}
		if source, ok := sources[name]; ok {
			pm.Source = &source
		}
		weightsStatus[name] = m.weightsStatus()
		switch weightsStatus[name] {
		case "hashing":
			printStatus(args, "rent_publish_weights_hashing", map[string]any{"model": name,
				"note": "emx is still hashing this model's weight files, so it is published without them and listed as Custom. Run lmx rent publish " + listing + " again once emx models shows them ready to have it verified."})
		case "error":
			printStatus(args, "rent_publish_weights_error", map[string]any{"model": name, "error": weights.Error,
				"note": "emx could not hash this model's weight files, so it is published without them and listed as Custom."})
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

	result := map[string]any{"listing": listing, "models": out, "weightsStatus": weightsStatus, "applied": false}
	scheduled := map[string]*rentScheduledPrice{}
	inForce := map[string]rentListingPrices{}
	if !hasFlag(args, "dry-run") {
		listed, err := rentPatchListingModels(cfg.Client, apiURL(args), key, listing, out)
		if err != nil {
			return err
		}
		for _, m := range listed {
			inForce[m.Model] = m
			if m.ScheduledPrice != nil {
				scheduled[m.Model] = m.ScheduledPrice
			}
		}
		result["applied"] = true
		result["scheduledPrices"] = scheduled
	}
	if hasFlag(args, "json") {
		printJSON(args, result)
		return nil
	}
	verb := "Listing " + listing + " now offers"
	if hasFlag(args, "dry-run") {
		verb = "Would set listing " + listing + " to offer"
	}
	fmt.Printf("%s (US dollars per 1M tokens):\n", verb)
	fmt.Printf("  %-32s %10s %10s %10s %9s  %-8s %s\n", "MODEL", "INPUT", "CACHED", "OUTPUT", "CONTEXT", "WEIGHTS", "SOURCE")
	for _, m := range out {
		// The table shows the prices in force: an increase waits for notice (listed below).
		price := rentListingPrices{PricePer1MInputTokens: m.PricePer1MInputTokens, PricePer1MCachedInputTokens: m.PricePer1MCachedInputTokens, PricePer1MOutputTokens: m.PricePer1MOutputTokens}
		if site, ok := inForce[m.Model]; ok {
			price = site
		}
		cached, ctx, source := "= input", "-", "-"
		if price.PricePer1MCachedInputTokens != nil {
			cached = usdPrice(*price.PricePer1MCachedInputTokens)
		}
		if m.ContextLength != nil {
			ctx = fmt.Sprint(*m.ContextLength)
		}
		if m.Source != nil {
			source = m.Source.Repo
			if m.Source.Revision != "" {
				source += "@" + m.Source.Revision[:min(12, len(m.Source.Revision))]
			}
		}
		fmt.Printf("  %-32s %10s %10s %10s %9s  %-8s %s\n", m.Model, usdPrice(price.PricePer1MInputTokens), cached, usdPrice(price.PricePer1MOutputTokens), ctx, weightsStatus[m.Model], source)
	}
	if len(scheduled) > 0 {
		names := make([]string, 0, len(scheduled))
		for name := range scheduled {
			names = append(names, name)
		}
		sort.Strings(names)
		fmt.Println("Price increases wait for notice: renters of this listing in the last 30 days were messaged; the prices above stay in force until then.")
		for _, name := range names {
			next := scheduled[name]
			at := next.EffectiveAt
			if t, err := time.Parse(time.RFC3339, at); err == nil {
				at = t.UTC().Format("2006-01-02 15:04 UTC")
			}
			cached := "= input"
			if next.PricePer1MCachedInputTokens != nil {
				cached = usdPrice(*next.PricePer1MCachedInputTokens)
			}
			fmt.Printf("  %-32s %10s %10s %10s  from %s\n", name, usdPrice(next.PricePer1MInputTokens), cached, usdPrice(next.PricePer1MOutputTokens), at)
		}
	}
	return nil
}

// usdPrice prints a price in US dollars per 1M tokens: 3 → "$3", 0.15 → "$0.15".
func usdPrice(v float64) string {
	return "$" + strconv.FormatFloat(v, 'f', -1, 64)
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
// rentScheduledPrice is a price increase the site holds back until effectiveAt, after notice to renters.
type rentScheduledPrice struct {
	EffectiveAt                 string   `json:"effectiveAt"`
	PricePer1MInputTokens       float64  `json:"pricePer1MInputTokens"`
	PricePer1MCachedInputTokens *float64 `json:"pricePer1MCachedInputTokens"`
	PricePer1MOutputTokens      float64  `json:"pricePer1MOutputTokens"`
}

// rentListingPrices is a listing model as the site answers it: the prices in force and any scheduled increase.
type rentListingPrices struct {
	Model                       string              `json:"model"`
	PricePer1MInputTokens       float64             `json:"pricePer1MInputTokens"`
	PricePer1MCachedInputTokens *float64            `json:"pricePer1MCachedInputTokens"`
	PricePer1MOutputTokens      float64             `json:"pricePer1MOutputTokens"`
	ScheduledPrice              *rentScheduledPrice `json:"scheduledPrice"`
}

// rentPatchListingModels sets the listing's models and returns them as the site now has them.
func rentPatchListingModels(client *http.Client, siteURL, key, listing string, models []rentPublishedModel) ([]rentListingPrices, error) {
	body, _ := json.Marshal(map[string]any{"models": models})
	ctx, cancel := context.WithTimeout(context.Background(), 30*time.Second)
	defer cancel()
	req, err := http.NewRequestWithContext(ctx, http.MethodPatch, strings.TrimRight(siteURL, "/")+"/api/rentals/"+listing, bytes.NewReader(body))
	if err != nil {
		return nil, err
	}
	req.Header.Set("Authorization", "Bearer "+key)
	req.Header.Set("Content-Type", "application/json")
	res, err := client.Do(req)
	if err != nil {
		return nil, cliError{"site_unreachable", "Could not reach " + siteURL + ": " + err.Error(), nil, nil}
	}
	defer res.Body.Close()
	answer, _ := io.ReadAll(io.LimitReader(res.Body, 1<<20))
	if res.StatusCode != http.StatusOK {
		return nil, cliError{"publish_failed", fmt.Sprintf("The site refused the update (HTTP %d): %s", res.StatusCode, strings.TrimSpace(string(answer))), []string{"The site allows one listing update per minute."}, nil}
	}
	var listed struct {
		Models []rentListingPrices `json:"models"`
	}
	_ = json.Unmarshal(answer, &listed) // older sites answer without models: nothing is scheduled then
	return listed.Models, nil
}
