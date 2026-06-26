#!/usr/bin/env bash
# Build the companion HellaSwag loglikelihood scorer (lmx-llama-score-hellaswag)
# and drop it next to the lmx binary. CPU-only, statically linked.
#
# Usage:
#   scripts/build-scorer.sh                       # fetch + build pinned llama.cpp
#   LLAMA_SRC=/path/to/llama.cpp scripts/build-scorer.sh   # reuse a local checkout
set -euo pipefail

here="$(cd "$(dirname "$0")/.." && pwd)"
src="$here/tools"
build="$src/build"

cmake_args=(-S "$src" -B "$build" -DCMAKE_BUILD_TYPE=Release)
if [[ -n "${LLAMA_SRC:-}" ]]; then
  cmake_args+=("-DFETCHCONTENT_SOURCE_DIR_LLAMA_CPP=$LLAMA_SRC")
fi

cmake "${cmake_args[@]}"
cmake --build "$build" --config Release -j "${JOBS:-4}"

cp "$build/lmx-llama-score-hellaswag" "$here/lmx-llama-score-hellaswag"
echo "built $here/lmx-llama-score-hellaswag"
