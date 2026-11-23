package main

import (
	"encoding/json"
	"fmt"
	"strconv"
	"strings"
	"time"
)

// rentHostPeriod is what a host's listings sold over a period (GET /api/rentals/host-stats).
type rentHostPeriod struct {
	InputTokens       int64   `json:"inputTokens"`
	CachedInputTokens int64   `json:"cachedInputTokens"`
	OutputTokens      int64   `json:"outputTokens"`
	Requests          int64   `json:"requests"`
	Renters           int64   `json:"renters"`
	GrossUSD          float64 `json:"grossUsd"`
	EarnedUSD         float64 `json:"earnedUsd"`
}

// rentHostStats is the site's answer to GET /api/rentals/host-stats.
type rentHostStats struct {
	AllTime     rentHostPeriod `json:"allTime"`
	Last30Days  rentHostPeriod `json:"last30Days"`
	Last24Hours rentHostPeriod `json:"last24Hours"`
	Models      []struct {
		ListingID    string         `json:"listingId"`
		ListingTitle string         `json:"listingTitle"`
		Model        string         `json:"model"`
		AllTime      rentHostPeriod `json:"allTime"`
		Last30Days   rentHostPeriod `json:"last30Days"`
		LastSoldAt   string         `json:"lastSoldAt"`
	} `json:"models"`
	Days []struct {
		Day          string  `json:"day"`
		InputTokens  int64   `json:"inputTokens"`
		OutputTokens int64   `json:"outputTokens"`
		EarnedUSD    float64 `json:"earnedUsd"`
	} `json:"days"`
	Earnings struct {
		AvailableUSD float64 `json:"availableUsd"`
		HeldUSD      float64 `json:"heldUsd"`
		PendingUSD   float64 `json:"pendingUsd"`
		PaidUSD      float64 `json:"paidUsd"`
	} `json:"earnings"`
}

// handleRentStats prints the account's hosting: tokens sold, requests and renters, what renters paid and
// what the host earned (all time, 30 days, 24 hours), each listing model, and where the earnings stand.
func handleRentStats(args cliArgs) error {
	key := apiKey(args)
	if key == "" {
		return missingAPIKey("lmx rent stats shows your listings' sales and earnings and needs your API key (a rental host key works).")
	}
	parsed, err := fetchJSON("GET", apiURL(args)+"/api/rentals/host-stats", key, nil)
	if err != nil {
		return err
	}
	if hasFlag(args, "json") && opt(args, "out") == "" {
		printJSON(args, parsed)
		return nil
	}
	if opt(args, "out") != "" {
		return writeOrPrintJSON("rent_stats", args, parsed)
	}
	raw, _ := json.Marshal(parsed)
	var stats rentHostStats
	if err := json.Unmarshal(raw, &stats); err != nil {
		return fmt.Errorf("unexpected answer from %s/api/rentals/host-stats: %w", apiURL(args), err)
	}
	if stats.AllTime.Requests == 0 {
		printInfo(args, "no_sales", map[string]any{"next": "Nothing sold yet: renters' billed requests to your listings show here (free listings and your own use are not counted)."})
		return nil
	}

	periods := []struct {
		name string
		p    rentHostPeriod
	}{{"all time", stats.AllTime}, {"30 days", stats.Last30Days}, {"24 hours", stats.Last24Hours}}
	rows := [][]string{}
	for _, period := range periods {
		p := period.p
		rows = append(rows, []string{
			period.name, groupThousands(p.InputTokens), groupThousands(p.CachedInputTokens), groupThousands(p.OutputTokens),
			groupThousands(p.Requests), groupThousands(p.Renters), usdAmount(p.GrossUSD), usdAmount(p.EarnedUSD),
		})
	}
	fmt.Print(asciiTable([]string{"PERIOD", "TOKENS IN", "OF WHICH CACHED", "TOKENS OUT", "REQUESTS", "RENTERS", "RENTERS PAID", "YOU EARNED"}, rows))

	e := stats.Earnings
	fmt.Printf("\nEarnings: %s payable now · %s held · %s in a payout · %s paid out\n\n",
		usdAmount(e.AvailableUSD), usdAmount(e.HeldUSD), usdAmount(e.PendingUSD), usdAmount(e.PaidUSD))

	rows = rows[:0]
	for _, m := range stats.Models {
		last := "-"
		if t, err := time.Parse(time.RFC3339, m.LastSoldAt); err == nil {
			last = t.Local().Format("2006-01-02")
		}
		rows = append(rows, []string{
			m.ListingID, truncateRentCell(m.ListingTitle, 32), m.Model, groupThousands(m.AllTime.Requests),
			groupThousands(m.AllTime.InputTokens), groupThousands(m.AllTime.OutputTokens),
			usdAmount(m.AllTime.EarnedUSD), usdAmount(m.Last30Days.EarnedUSD), last,
		})
	}
	fmt.Print(asciiTable([]string{"LISTING", "TITLE", "MODEL", "REQUESTS", "TOKENS IN", "TOKENS OUT", "EARNED", "EARNED 30D", "LAST SOLD"}, rows))
	fmt.Println("\nBilled requests only (free listings and your own use are not counted); earned is your share after the platform fee.")
	return nil
}

// groupThousands writes n with thousands separators: 1234567 -> 1,234,567.
func groupThousands(n int64) string {
	s := strconv.FormatInt(n, 10)
	neg := strings.HasPrefix(s, "-")
	s = strings.TrimPrefix(s, "-")
	var b strings.Builder
	for i, r := range s {
		if i > 0 && (len(s)-i)%3 == 0 {
			b.WriteByte(',')
		}
		b.WriteRune(r)
	}
	if neg {
		return "-" + b.String()
	}
	return b.String()
}

// usdAmount writes US dollars to the cent, or with two significant digits below a cent ($0.0037).
func usdAmount(v float64) string {
	if v == 0 || v >= 0.01 || v <= -0.01 {
		return "$" + strconv.FormatFloat(v, 'f', 2, 64)
	}
	return "$" + strconv.FormatFloat(v, 'g', 2, 64)
}
