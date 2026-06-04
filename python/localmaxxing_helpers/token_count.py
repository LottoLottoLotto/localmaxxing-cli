#!/usr/bin/env python3
"""Optional tokenizer helper for the Go LocalMaxxing CLI.

The Go CLI should call this only for best-effort ML ecosystem behavior. It reads
JSON on stdin: {"model":"Qwen/Qwen3-8B", "text":"...", "revision":"main"}
and writes {"tokens": 123}.
"""

from __future__ import annotations

import json
import sys


def main() -> int:
    request = json.load(sys.stdin)
    model = request.get("model")
    text = request.get("text", "")
    revision = request.get("revision") or None
    if not model:
        print(json.dumps({"error": "model is required"}), file=sys.stderr)
        return 2

    try:
        from transformers import AutoTokenizer
    except Exception as exc:
        print(json.dumps({"error": "transformers is not installed", "details": str(exc)}), file=sys.stderr)
        return 3

    tokenizer = AutoTokenizer.from_pretrained(model, revision=revision, trust_remote_code=True)
    encoded = tokenizer.encode(text, add_special_tokens=False)
    print(json.dumps({"tokens": len(encoded)}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
