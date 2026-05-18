#!/usr/bin/env bash
set -euo pipefail

APP_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
XCODEGEN_VERSION="2.45.4"

cd "$APP_DIR"

"$APP_DIR/scripts/fetch_mediapipe_ios.sh"
"$APP_DIR/scripts/fetch_sherpa_onnx_ios.sh"

if command -v xcodegen >/dev/null 2>&1; then
  xcodegen generate --spec project.yml
else
  XCODEGEN_DIR="${TMPDIR:-/tmp}/xcodegen-${XCODEGEN_VERSION}"
  if [ ! -d "$XCODEGEN_DIR/.git" ]; then
    git clone --depth 1 --branch "$XCODEGEN_VERSION" https://github.com/yonaskolb/XcodeGen.git "$XCODEGEN_DIR"
  fi

  cd "$XCODEGEN_DIR"
  swift run xcodegen generate --spec "$APP_DIR/project.yml" --project "$APP_DIR"
  cd "$APP_DIR"
fi

rm -rf "$APP_DIR/GemmaPi.xcworkspace"
