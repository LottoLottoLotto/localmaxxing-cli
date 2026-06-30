# Hardware and setups

## Detect hardware

Run hardware detection on the machine that actually serves the model. For remote benchmarks, that means the server, not the client running `lmx`.

```bash
lmx hardware --out hardware.json
lmx hardware init --out hardware.json
```

## Hardware templates

Generate a hardware object from explicit flags when automatic detection is unavailable:

```bash
lmx hardware template \
  --gpu-name "RTX 3090" \
  --gpu-count 2 \
  --vram-gb 24 \
  --cpu "Ryzen 9 9950X" \
  --ram-gb 96 \
  --os Linux \
  --out hardware.json
```

Hardware classes:

- `DISCRETE_GPU`: use `gpuName`, `gpuCount`, and `vramGb`; mixed GPU systems can use `gpus[]`.
- `UNIFIED`: use `chipVendor`, `chipFamily`, and `unifiedMemoryGb`.
- `CPU_ONLY`: use CPU/system metadata without GPU fields.

Common fields include `cpu`, `ramGb`, `os`, and `powerWatts`.

## Saved setups

Saved setups require an API key. List or pull account setups:

```bash
lmx setups list
lmx setups pull --default --out hardware.json
lmx setups pull --name "2x RTX 3090" --out hardware.json
lmx setups pull --id <id> --out hardware.json
```

Selection flags:

- `--default`: pull the default saved setup.
- `--name <name>`: pull by saved setup name, case-insensitive.
- `--id <id>`: pull by saved setup id.
