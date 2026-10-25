package main

import (
	"context"
	"strings"
	"time"
)

// canonicalPrompt mirrors lib/canonicalPrompts.ts on localmaxxing.com. A
// speed-test run only earns the "Verified run" badge when it was measured
// against one of these prompts, so the text here must match the server
// byte-for-byte. Never edit an existing prompt; add a new id instead.
type canonicalPrompt struct {
	ID              string
	Title           string
	MinOutputTokens int
	Text            string
}

var canonicalPrompts = []canonicalPrompt{
	{
		ID:              "reasoning-v1",
		Title:           "Mixed reasoning",
		MinOutputTokens: 256,
		Text: strings.Join([]string{
			"You are helping a small robotics club plan a weekend build. Work through each part carefully and show your reasoning.",
			"",
			"1. The club has 7 members and a budget of 412 dollars. A motor driver costs 18.50, a microcontroller 23.75, and each member wants one of each plus a shared 3D-printer spool at 27.90. After buying everything, they split the remainder evenly on batteries priced at 6.35 each. How many batteries can they buy in total, and how much money is left over?",
			"",
			"2. Two members disagree about wheel size. Alice says a larger wheel gives more torque at the ground; Ben says it gives more speed. Explain who is right, what tradeoff is actually involved, and how gear ratio changes the answer.",
			"",
			"3. Write a short, plain-English procedure (six to nine steps) for safely testing a new motor circuit for the first time, including what to check before applying power.",
			"",
			"4. Finally, propose one creative feature for the robot that would be unusual for a club project, and give two concrete risks with how you would mitigate each.",
			"",
			"Answer all four parts in order with clear headings.",
		}, "\n"),
	},
	{
		ID:              "code-v1",
		Title:           "Code generation",
		MinOutputTokens: 256,
		Text: strings.Join([]string{
			"Implement a small command-line tool in Python called `logtally`.",
			"",
			"Requirements:",
			"- It reads newline-delimited JSON log records from stdin. Each record has `ts` (ISO-8601 string), `level` (one of DEBUG, INFO, WARN, ERROR), `service` (string), and optional `latency_ms` (number).",
			"- It prints, per service, the count of records by level and the p50 and p95 of `latency_ms` (ignore records without latency).",
			"- Malformed lines must be counted and reported at the end, never crash the program.",
			"- Support a `--since <ISO timestamp>` flag that drops earlier records.",
			"- Use only the standard library.",
			"",
			"After the code, explain in two or three sentences how you computed percentiles and why you chose that method, then list three edge cases a unit test suite should cover.",
		}, "\n"),
	},
}

func embeddedCanonicalPrompt(id string) (canonicalPrompt, bool) {
	for _, prompt := range canonicalPrompts {
		if prompt.ID == id {
			return prompt, true
		}
	}
	return canonicalPrompt{}, false
}

func canonicalPromptIDs(prompts []canonicalPrompt) []string {
	ids := make([]string, 0, len(prompts))
	for _, prompt := range prompts {
		ids = append(ids, prompt.ID)
	}
	return ids
}

// fetchCanonicalPrompts loads the live canonical prompt list from
// GET /api/agent-context so a CLI build that predates a new prompt id can
// still run it. Any failure yields nil; callers fall back to the embedded table.
func fetchCanonicalPrompts(args cliArgs) []canonicalPrompt {
	ctx, cancel := context.WithTimeout(context.Background(), 5*time.Second)
	defer cancel()
	value, err := fetchJSONContext(ctx, "GET", apiURL(args)+"/api/agent-context", "", nil)
	if err != nil {
		return nil
	}
	items := anySlice(asObject(value)["canonicalPrompts"])
	if len(items) == 0 {
		return nil
	}
	prompts := make([]canonicalPrompt, 0, len(items))
	for _, item := range items {
		obj := asObject(item)
		id := stringValue(obj["id"])
		text := stringValue(obj["text"])
		if obj == nil || id == "" || text == "" {
			continue
		}
		if sha := stringValue(obj["sha256"]); sha != "" && sha != sha256Hex(text) {
			continue
		}
		prompts = append(prompts, canonicalPrompt{ID: id, Title: stringValue(obj["title"]), MinOutputTokens: int(numberField(obj, "minOutputTokens")), Text: text})
	}
	if len(prompts) == 0 {
		return nil
	}
	return prompts
}

// resolveCanonicalPrompt returns the prompt for --canonical-prompt <id>,
// preferring the server's copy and falling back to the embedded table.
func resolveCanonicalPrompt(args cliArgs, id string) (canonicalPrompt, error) {
	embedded, haveEmbedded := embeddedCanonicalPrompt(id)
	source := "embedded"
	selected := embedded
	found := haveEmbedded
	if remote := fetchCanonicalPrompts(args); remote != nil {
		for _, prompt := range remote {
			if prompt.ID != id {
				continue
			}
			selected = prompt
			found = true
			source = "server"
			if !haveEmbedded {
				source = "server_only"
			} else if prompt.Text == embedded.Text {
				source = "embedded_matches_server"
			}
			break
		}
	}
	if !found {
		return canonicalPrompt{}, cliError{"unknown_canonical_prompt", "Unknown canonical prompt id: " + id, []string{"Use one of: " + strings.Join(canonicalPromptIDs(canonicalPrompts), ", ") + ".", "Run lmx context get canonicalPrompts to list the server's current prompts."}, nil}
	}
	printStatus(args, "canonical_prompt_source", map[string]any{"id": selected.ID, "source": source, "sha256": sha256Hex(selected.Text)})
	return selected, nil
}
