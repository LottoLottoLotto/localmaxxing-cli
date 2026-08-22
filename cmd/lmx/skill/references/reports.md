# Model reports

Model reports use one canonical document format across the API, CLI, and web report studio: GitHub Flavored Markdown (`contentFormat: "gfm"`, format version `1`). Raw HTML is not rendered.

## Discover the live contract

Always fetch the live contract before generating a report when compatibility matters:

```bash
lmx report format --json
```

It returns supported syntax, Localmaxxing tokens, limits, submission/editing endpoints, and an example payload.

## Author a report

Generate a structured template:

```bash
lmx report init --out report.md
```

Use standard GFM:

```markdown
## Setup

Ran **Qwen** with `llama.cpp`.

- GPU: RTX 3090
- Quant: Q4_K_M

## Results

| Metric | Value |
| --- | ---: |
| Decode | 42.5 tok/s |
```

Create a public report:

```bash
lmx report create \
  --model Qwen/Qwen3.8-27B \
  --title "Qwen local inference report" \
  --summary "Reproducible setup, measurements, and observations." \
  --content-file report.md \
  --json
```

Create a private draft for browser review by adding `--draft`. The returned `id` is the canonical report ID. A user can open the report in the web studio and edit the exact same Markdown.

Attach owned runs with comma-separated IDs:

```bash
lmx report create ... \
  --benchmark-run-ids runA,runB \
  --eval-run-ids evalA
```

## Read and edit

```bash
lmx report list --model Qwen/Qwen3.8-27B --json
lmx report show <reportId> --json
lmx report edit <reportId> --content-file revised.md --json
lmx report edit <reportId> --title "Revised title" --summary "Revised summary with enough detail."
```

Omitting evidence flags preserves current attachments. Passing an explicitly empty flag, such as `--benchmark-run-ids=`, clears that attachment list.

Publication state is independent from editing:

```bash
lmx report publish <reportId>
lmx report unpublish <reportId>
lmx report delete <reportId> --yes
```

## Inline images

Upload evidence after creating the report:

```bash
lmx report image upload <reportId> \
  --file evidence.png \
  --caption "Measured output" \
  --sort-order 0 \
  --json
```

Insert the returned image `id` into the Markdown at the desired location:

```text
[[report-image:<image-id>]]
```

Then save the content with `lmx report edit`. Delete an image with:

```bash
lmx report image delete <reportId> <imageId> --yes
```

Embed GitHub, Hugging Face, arXiv, X, or Localmaxxing sources using a percent-encoded absolute URL:

```text
[[report-embed:https%3A%2F%2Fgithub.com%2Fowner%2Frepository]]
```

## Agent invariants

- Use `--json` when consuming output programmatically.
- Prefer `--content-file` or `--content-file -` over shell-escaped multiline `--content`.
- Never emit raw HTML; it is intentionally ignored.
- Do not invent image IDs or evidence run IDs.
- Create with `--draft` when a human must review before publication.
- Editing through CLI, API, or web always updates the same GFM content; no conversion step is required.
