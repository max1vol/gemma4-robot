#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
IMAGE="${POSE_AARCH64_IMAGE:-gemma4-xnnpack-build:bookworm-arm64}"
OUT="${1:-out/pose_neon_runtime_aarch64_ofast}"

cd "$ROOT"

container run --rm --arch arm64 \
  -v "$ROOT:/work" \
  -w /work \
  "$IMAGE" \
  /bin/bash -lc "gcc -Ofast -mcpu=cortex-a53 -DPOSE_USE_A53_PW6X8_ASM -std=c11 -Wall -Wextra -I out/pose_runtime_data pose_estimation/pose_runtime.c pose_estimation/a53_pw6x8.S -o '$OUT' -lm -pthread"
