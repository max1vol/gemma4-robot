#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
IMAGE="${IMAGE:-rust:1-bookworm}"
CARGO_CACHE="${CARGO_CACHE:-$ROOT/.cargo-container}"

if ! command -v container >/dev/null 2>&1; then
  echo "Apple container CLI not found: install/use the macOS container runtime on this laptop." >&2
  exit 1
fi

mkdir -p "$CARGO_CACHE"

container run --rm \
  --uid "$(id -u)" \
  --gid "$(id -g)" \
  --env CARGO_HOME=/work/.cargo-container \
  --env PATH=/usr/local/cargo/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin \
  --mount "type=bind,source=${ROOT},target=/work" \
  --workdir /work/agent-harness \
  "$IMAGE" \
  sh -c 'if [ ! -f Cargo.lock ]; then cargo generate-lockfile; fi; cargo build --release --locked'

mkdir -p "$ROOT/bin"
cp "$ROOT/agent-harness/target/release/gemma-agent-harness" "$ROOT/bin/gemma-agent-harness"
chmod +x "$ROOT/bin/gemma-agent-harness"
echo "$ROOT/bin/gemma-agent-harness"
