# Host a model on LocalMaxxing

Rent out a model running on your own machine. Renters call it through
LocalMaxxing with an OpenAI-compatible API and pay per token; you keep 90% of
what they pay. Your machine opens **no ports**: `lmx rent host` makes one
outbound connection to LocalMaxxing's relay, and renter requests come back down
it to your local EndpointMaxxing (`emx`) gateway.

```
renter ──HTTPS──▶ localmaxxing.com ◀──one outbound WebSocket (443)── lmx rent host ──localhost──▶ emx ──▶ engine (llama.cpp, vLLM, …)
```

This guide sets up a host from nothing, in order. Each step ends with a check;
do not go on until it passes. Commands are for Linux (systemd); macOS
differences are noted. Agents: follow it top to bottom, substituting the
values in **Choices** below, and never print secrets (keys) to the terminal or
chat; write them to files with `umask 077` as shown.

## What you need

- A **LocalMaxxing Pro** account (hosting needs Pro): <https://www.localmaxxing.com/pro>.
- A machine that stays on, with the model's memory free (a GPU, Apple Silicon,
  or CPU for small models), and outbound HTTPS (port 443) to
  `www.localmaxxing.com`. No public IP, port forwarding or tunnel.
- `emx` **v0.3.4 or later** and `lmx` **v0.1.65 or later**.

## Choices

Decide these first; the commands below use these names.

| Name | Example | Notes |
|---|---|---|
| `MODEL` | `qwen3-0.6b-q8` | The name renters use. Letters, digits, `._:/@+-`. |
| Model files | `unsloth/Qwen3-0.6B-GGUF`, file `Qwen3-0.6B-Q8_0.gguf` | Download them **unmodified** from Hugging Face if you want **Verified** weights (see step 7). |
| Engine | `llama.cpp` (GGUF), `vllm` / `sglang` (safetensors) | emx picks one that is installed when you do not say. |
| Price | `$0.10` in / `$0.40` out per 1M tokens | US dollars per million tokens; cached input optional (default: the input price). |
| Concurrency | `4` | Requests served at once; more queue. |
| Rentals home | `~/emx-rentals` | A separate emx just for rentals, so your own emx setup stays as it is. |
| Port | `7810` | Where the rentals emx listens, on `127.0.0.1` only. |

## 1. Install emx and lmx

```sh
# emx: https://github.com/LottoLottoLotto/endpointmaxxing/releases (pick your OS/arch)
v=$(curl -fsSL https://api.github.com/repos/LottoLottoLotto/endpointmaxxing/releases/latest | sed -n 's/.*"tag_name": "v\([^"]*\)".*/\1/p')
curl -fsSLO "https://github.com/LottoLottoLotto/endpointmaxxing/releases/download/v$v/emx_${v}_linux_amd64.tar.gz"
curl -fsSLO "https://github.com/LottoLottoLotto/endpointmaxxing/releases/download/v$v/checksums.txt"
sha256sum -c checksums.txt --ignore-missing && tar -xzf "emx_${v}_linux_amd64.tar.gz" && install -m 0755 emx ~/.local/bin/emx

# lmx: https://github.com/LottoLottoLotto/localmaxxing-cli/releases
base=https://github.com/LottoLottoLotto/localmaxxing-cli/releases/latest/download
curl -fsSLO "$base/lmx-linux-amd64.tar.gz" && curl -fsSLO "$base/checksums.txt"
sha256sum -c checksums.txt --ignore-missing && tar -xzf lmx-linux-amd64.tar.gz && install -m 0755 lmx ~/.local/bin/lmx
```

macOS: use the `darwin_arm64` / `darwin-arm64` archives and
`shasum -a 256 -c checksums.txt --ignore-missing`.

**Check:** `emx version` and `lmx version` print at least v0.3.4 and v0.1.65;
`EMX_HOME=~/emx-rentals/home emx doctor` (after step 2) lists the engine you
chose as installed (`emx engine install <id>` installs a missing one).

Pin the engine's binary to an absolute path once step 2 exists: services
started by systemd have a different `PATH` from your shell, and may find
another (or a broken) copy:

```sh
EMX_HOME=~/emx-rentals/home emx engine set llama.cpp --bin "$(command -v llama-server)"
```

## 2. Create the rentals emx

Everything rentals-related lives in its own emx home, with a gateway that
listens on `127.0.0.1` only, keeps renters' prompt caches apart, and restarts
crashed engines.

```sh
export EMX_HOME=~/emx-rentals/home        # use this in every emx command below
mkdir -p "$EMX_HOME/config"
( umask 077 && openssl rand -hex 32 > ~/emx-rentals/master.key )   # the gateway's admin key
cat > "$EMX_HOME/config/config.yaml" <<'EOF'
gateway:
    host: 127.0.0.1
    port: 7810
    autostart: true          # load a model on its first request
    placement: memory
    restart_crashed: true
    cache_isolation: true    # each renter gets their own prompt cache: required for rentals
    stream_usage: true
    max_queue: 64
logging:
    enabled: true
    bodies: false            # do not store prompts
EOF
```

Never point `EMX_HOME` at a directory other people can read: it holds keys.

## 3. Download and register the model

```sh
# Unmodified files from Hugging Face (pin the commit so the weights can be verified later):
lmx model pull unsloth/Qwen3-0.6B-GGUF --file Qwen3-0.6B-Q8_0.gguf --dry-run
#   JSON: source.revision is the commit, destination the path it downloads to
lmx model pull unsloth/Qwen3-0.6B-GGUF --file Qwen3-0.6B-Q8_0.gguf --revision <commit> --yes

EMX_HOME=~/emx-rentals/home emx model add qwen3-0.6b-q8 <destination> --ctx 32768 --parallel 4
```

Useful `emx model add` options: `--engine ID` to pin an engine, `--ctx N`
context (llama.cpp splits it over the slots: `--ctx 32768 --parallel 4` gives
each request 8192 tokens, which is what renters see), `--parallel N` request
slots (match your concurrency),
`--gpus 0` / `--gpu-layers all` / `--mem-fraction 0.9` for placement. A custom
launcher (your own vLLM script, for example) is added as an engine manifest
with `emx engine add file.yaml --trust`; make sure it listens on the `{host}`
emx gives it, not `0.0.0.0` (otherwise anyone on your network can use it
without a key).

**Check:** `EMX_HOME=~/emx-rentals/home emx models` shows the model with
status `ok`.

## 4. Set the price

```sh
EMX_HOME=~/emx-rentals/home emx price set qwen3-0.6b-q8 --prompt 0.10 --completion 0.40   # optional: --cached 0.01
```

US dollars per 1M tokens. LocalMaxxing never offers a model at a price you did
not set. Lowering a price later takes effect at once; raising it waits 24
hours after renters are told.

**Check:** `EMX_HOME=~/emx-rentals/home emx price list` shows it.

## 5. Run the gateway as a service

```sh
mkdir -p ~/.config/systemd/user
cat > ~/.config/systemd/user/emx-rentals.service <<'EOF'
[Unit]
Description=EndpointMaxxing gateway for LocalMaxxing rentals
After=network-online.target

[Service]
Environment=EMX_HOME=%h/emx-rentals/home
ExecStart=%h/.local/bin/emx serve --api-key-file %h/emx-rentals/master.key
Restart=on-failure
RestartSec=5
KillMode=mixed
TimeoutStopSec=90

[Install]
WantedBy=default.target
EOF
systemctl --user daemon-reload
systemctl --user enable --now emx-rentals
loginctl enable-linger "$USER"      # keep user services running after logout and start them at boot
```

macOS: run the same command from a `launchd` agent (a plist in
`~/Library/LaunchAgents` with `RunAtLoad` and `KeepAlive`, and `EMX_HOME` in
`EnvironmentVariables`).

**Check:** load the model and get an answer (the first request starts it,
which can take minutes for large models):

```sh
curl -s http://127.0.0.1:7810/v1/chat/completions -H "Authorization: Bearer $(cat ~/emx-rentals/master.key)" \
  -H 'content-type: application/json' \
  -d '{"model":"qwen3-0.6b-q8","messages":[{"role":"user","content":"Say hello."}],"max_tokens":20}'
```

An answer means the model works. `journalctl --user -u emx-rentals` shows the
gateway's log, `EMX_HOME=~/emx-rentals/home emx logs <instance>` the engine's.

## 6. Make the key lmx uses to call emx

lmx calls your gateway with its own key, limited to the rented models. It is a
**delegated** key: emx records each renter separately and gives each their own
prompt cache.

```sh
( umask 077 && EMX_HOME=~/emx-rentals/home emx key add lmx-rentals --delegate --models qwen3-0.6b-q8 \
    | grep -o 'sk-emx-[0-9a-f]*' > ~/emx-rentals/lmx-target.key )
```

The key is shown only once; this writes it straight to a file instead of the
terminal. Add more models later with `emx key set lmx-rentals --models a,b`.

## 7. Create the listing and a rental host key

On <https://www.localmaxxing.com>:

1. **Rentals → List your endpoint**: title, description, hardware, the model
   label, max concurrency, public or private. Save it as a draft and note its
   **listing id** (in its URL, `cm…`).
2. **Dashboard → API Keys → New Key**, tick **Rental host key**, and save it
   to a file on the host machine:

   ```sh
   ( umask 077 && printf 'LMX_API_KEY=%s\nLMX_RENT_TARGET_KEY=%s\n' '<the bhk_ key>' "$(cat ~/emx-rentals/lmx-target.key)" > ~/emx-rentals/host.env )
   ```

   A rental host key can only connect `lmx rent host`, list your listings and
   publish their models. Do not use your full account key on a host machine.

Agents with a **full** API key can create the listing through the API
instead (`tags` is a JSON array encoded as a string):

```sh
curl -s https://www.localmaxxing.com/api/rentals -H "Authorization: Bearer $FULL_KEY" -H 'content-type: application/json' -d '{
  "title": "Qwen3 0.6B on my RTX 3090", "description": "…", "hardwareLabel": "1x RTX 3090 24 GB",
  "maxConcurrency": 4, "accessMode": "PUBLIC", "tags": "[\"qwen\",\"fast\"]",
  "models": [{"model": "qwen3-0.6b-q8", "pricePer1MInputTokens": 0.10, "pricePer1MOutputTokens": 0.40}]}'
```

## 8. Publish the model, prices and weights

`lmx rent publish` reads the model, its context length, your emx price and
emx's fingerprint of the weight files from the gateway, and puts them on the
listing. It reads prices with the gateway's **admin** key:

```sh
set -a; . ~/emx-rentals/host.env; set +a
EMX_API_KEY=$(cat ~/emx-rentals/master.key) lmx rent publish <listingId> \
  --target http://127.0.0.1:7810 --model qwen3-0.6b-q8 \
  --source qwen3-0.6b-q8=unsloth/Qwen3-0.6B-GGUF@<commit>
```

Expected: a table with the model, your prices, its context, `WEIGHTS ready`
and the source. The site then checks the files against Hugging Face:

- **Verified**: your files are exactly the repo's at that commit. Verified
  public listings join the model's **pool** (renters can ask for the model and
  get the cheapest host) and get hidden spot checks that the answers come from
  those weights.
- **Custom**: no `--source`, or files that are not byte-for-byte the repo's
  (requantized, merged, fine-tuned). The listing works normally but joins no
  pool. Say what the model is in the description.
- **Mismatch**: the files differ from the repo you named. Fix the source or the
  files and publish again.

If emx is still hashing a big model, publish waits up to `--wait-hashing`
(default 120s); run it again later if it says `hashing`. A listing takes one
change per minute: wait a minute after publishing before the next change
(activating it, for example).

## 9. Activate the listing and start hosting

Set the listing to **ACTIVE** on its edit page (a rental host key cannot
change status). Then run the host agent as a service:

```sh
cat > ~/.config/systemd/user/lmx-rent-host.service <<'EOF'
[Unit]
Description=LocalMaxxing rentals host agent
After=emx-rentals.service network-online.target
Wants=emx-rentals.service

[Service]
EnvironmentFile=%h/emx-rentals/host.env
ExecStart=%h/.local/bin/lmx rent host <listingId> --target http://127.0.0.1:7810 --model qwen3-0.6b-q8 --max-concurrency 4
Restart=always
RestartSec=5

[Install]
WantedBy=default.target
EOF
systemctl --user daemon-reload && systemctl --user enable --now lmx-rent-host
journalctl --user -u lmx-rent-host -n 20 --no-pager
```

**Check:** the log shows `rent_host_connected` with your listing `(ACTIVE)`,
and

```sh
curl -s https://www.localmaxxing.com/api/rentals/<listingId>/availability
```

answers `"hostOnline":true`. Several listings can share one agent: list their
ids after `rent host`.

## 10. Test it like a renter

Call your listing with your **full** account key (owners using their own
listing are not charged):

```sh
curl -s https://www.localmaxxing.com/api/rentals/<listingId>/v1/chat/completions -H "Authorization: Bearer $FULL_KEY" \
  -H 'content-type: application/json' -d '{"model":"qwen3-0.6b-q8","messages":[{"role":"user","content":"Hi"}],"max_tokens":20}'
```

Renters use `https://www.localmaxxing.com/api/rentals/<listingId>/v1` as their
OpenAI base URL (or `lmx rent connect <listingId>` for a local one). Thinking
models: renters send `"reasoning_effort": "none"` to turn thinking off.

## 11. Get paid

**Dashboard → Credits & payouts → Connect payouts** sets up a Stripe payout
account (you can do it before earning anything). Renters pay in US dollars; a
USD bank account avoids conversion fees. Earnings are held 14 days, then
**Request payout** once at least $25 is payable; Stripe pays monthly.

See what you sold: **Dashboard → Rentals → Your hosting**, or

```sh
set -a; . ~/emx-rentals/host.env; set +a; lmx rent stats
```

emx keeps its own per-model numbers on the machine:
`EMX_HOME=~/emx-rentals/home emx usage` / `emx report`.

## Running it day to day

| Task | Command |
|---|---|
| Status | `systemctl --user status emx-rentals lmx-rent-host` |
| Logs | `journalctl --user -u lmx-rent-host -f`, `journalctl --user -u emx-rentals -f` |
| Stop hosting (listing shows offline) | `systemctl --user stop lmx-rent-host` |
| Change a price | `emx price set …`, then `lmx rent publish …` again |
| Add a model | `emx model add …`, `emx price set …`, `emx key set lmx-rentals --models a,b`, `lmx rent publish … --model a --model b`, add `--model b` to the service and restart it |
| Update | `lmx update`; new emx release as in step 1; then `systemctl --user restart emx-rentals lmx-rent-host` |

Keep the machine for the listing: another program taking the GPU's memory
makes requests fail, and the listing gets benched after repeated failures.

## What renters can and cannot reach

- Only `POST /v1/chat/completions`, `/v1/completions`, `/v1/embeddings` and
  `GET /v1/models`, only the models you publish; `lmx` enforces this on your
  machine, and its emx key only works for those models.
- Renters never see your IP address. You **can** see what they send (the model
  runs on your machine): do not log or keep prompts (`logging.bodies: false`).
- A request that produces no output for 150 s is cancelled; renters whose
  requests keep doing that are blocked from your listing for a while.

## Troubleshooting

| Message | Meaning and fix |
|---|---|
| `not_your_listing` | The key in `host.env` belongs to another account, or the listing id is wrong (`lmx rent list --mine`). |
| `model_unpriced` | No `emx price` for the model, or `lmx rent publish` ran without the gateway's admin key in `EMX_API_KEY`. |
| `rent_host_weights_changed` | The files emx serves differ from the ones published; run `lmx rent publish` again. |
| Listing offline although the service runs | Look for errors in `journalctl --user -u lmx-rent-host`; a 401 from the relay means the key was deleted or the account is not Pro. |
| Requests fail with 502/503 | emx cannot start or reach the model: `journalctl --user -u emx-rentals`, `emx logs`, free GPU memory. |
| `exited during startup; log: …` | Read that log. `error while loading shared libraries` or a wrong version means the service found another engine binary: pin it with `emx engine set <engine> --bin /absolute/path`. |
| `not enough GPU memory` | Something else holds the GPU (another emx, a game, a notebook); free it, or lower `--ctx` / `--mem-fraction`. |
| `Rate limit exceeded. Please wait before updating again.` | One change per listing per minute; try again in a minute. |
| `listing models` / `403` on publish with the site | A rental host key only publishes models; change title, status and access on the site. |
| Weights stay `Custom` | No `--source`, or the files are not the repo's exact files at that commit. |
| `Your last 3 requests … produced no output` (renters) | Your engine produced nothing for 150 s; check its log and memory. |

Full command reference: `lmx rent host --help`, `lmx rent publish --help`, and
the **Rentals** section of the lmx README.
