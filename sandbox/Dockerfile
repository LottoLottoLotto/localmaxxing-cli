# Hardened sandbox image for execution-based code eval shards (HumanEval / MBPP).
#
# Build:
#   docker build -t lmx-sandbox sandbox
#
# The default `lmx eval shard <code-dataset>` launcher uses:
#   docker run --rm -i \
#     --network none \
#     --memory 2g --cpus 2 --pids-limit 128 \
#     --cap-drop ALL --security-opt no-new-privileges \
#     --read-only --tmpfs /tmp:exec,size=128m \
#     lmx-sandbox
# Hosts whose Docker/rootless/security profile rejects the strict exec profile can
# keep the network/resource limits but omit cap/no-new-privileges/read-only with
# `--sandbox-relaxed-security`.
#
# Tasks arrive as JSONL on stdin; results leave as JSONL on stdout. Only the
# Python standard library is available (HumanEval/MBPP need nothing else), so the
# attack surface and dependency drift are minimal.
FROM python:3.12-slim

# Non-root execution user with no writable home; root filesystem is mounted
# read-only at runtime, /tmp is a tmpfs.
RUN useradd --create-home --shell /usr/sbin/nologin sandbox

WORKDIR /sandbox
COPY run_sandbox.py /sandbox/run_sandbox.py

# EvalPlus (HumanEval+/MBPP+) test harnesses import numpy; install it so the
# verifier can run the canonical extended tests.
RUN pip install --no-cache-dir numpy==2.1.3

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    SANDBOX_CPU_SECONDS=20 \
    SANDBOX_WALL_SECONDS=45 \
    SANDBOX_MEM_MB=2048

USER sandbox

ENTRYPOINT ["python3", "/sandbox/run_sandbox.py"]
