#!/usr/bin/env python3
"""Optional tokenizer helper for the Go LocalMaxxing CLI.

Reads JSON on stdin. Supported requests:
- {"model":"Qwen/Qwen3-8B", "text":"...", "revision":"main"}
  -> {"tokens": 123}
- {"model":"Qwen/Qwen3-8B", "target_tokens": 10000, "seed_text":"..."}
  -> {"tokens": 10000, "text":"..."}
"""

from __future__ import annotations

import json
import sys


def load_tokenizer(model: str, revision: str | None):
    try:
        from transformers import AutoTokenizer
    except Exception as exc:
        print(json.dumps({"error": "transformers is not installed", "details": str(exc)}), file=sys.stderr)
        raise SystemExit(3)
    return AutoTokenizer.from_pretrained(model, revision=revision, trust_remote_code=True)


def exact_token_text(tokenizer, seed_text: str, target_tokens: int) -> str:
    if target_tokens <= 0:
        raise ValueError("target_tokens must be positive")
    seed_text = (seed_text or "context ").strip() or "context"
    text = seed_text
    ids = tokenizer.encode(text, add_special_tokens=False)
    while len(ids) < target_tokens:
        text = text + " " + seed_text
        ids = tokenizer.encode(text, add_special_tokens=False)
    candidate = tokenizer.decode(ids[:target_tokens], skip_special_tokens=False, clean_up_tokenization_spaces=False)
    verified = tokenizer.encode(candidate, add_special_tokens=False)
    if len(verified) != target_tokens:
        raise ValueError(f"decoded filler re-encoded to {len(verified)} tokens, wanted {target_tokens}")
    return candidate


def main() -> int:
    request = json.load(sys.stdin)
    model = request.get("model")
    revision = request.get("revision") or None
    if not model:
        print(json.dumps({"error": "model is required"}), file=sys.stderr)
        return 2

    tokenizer = load_tokenizer(model, revision)
    if "target_tokens" in request:
        target_tokens = int(request.get("target_tokens") or 0)
        text = exact_token_text(tokenizer, request.get("seed_text", ""), target_tokens)
        print(json.dumps({"tokens": target_tokens, "text": text}))
        return 0

    text = request.get("text", "")
    encoded = tokenizer.encode(text, add_special_tokens=False)
    print(json.dumps({"tokens": len(encoded)}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
