#!/usr/bin/env bash
set -euo pipefail

APP_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
XCODEGEN_VERSION="2.45.4"

cd "$APP_DIR"

if command -v xcodegen >/dev/null 2>&1; then
  xcodegen generate --spec project.yml
  exit 0
fi

XCODEGEN_DIR="${TMPDIR:-/tmp}/xcodegen-${XCODEGEN_VERSION}"
if [ ! -d "$XCODEGEN_DIR/.git" ]; then
  git clone --depth 1 --branch "$XCODEGEN_VERSION" https://github.com/yonaskolb/XcodeGen.git "$XCODEGEN_DIR"
fi

cd "$XCODEGEN_DIR"
swift run xcodegen generate --spec "$APP_DIR/project.yml" --project "$APP_DIR"
