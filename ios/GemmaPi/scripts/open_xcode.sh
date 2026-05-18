#!/usr/bin/env bash
set -euo pipefail

APP_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

"$APP_DIR/scripts/generate_project.sh"

if ! command -v xed >/dev/null 2>&1; then
  echo "Missing required tool: xed. Open $APP_DIR/GemmaPi.xcodeproj in Xcode." >&2
  exit 1
fi

xed "$APP_DIR/GemmaPi.xcodeproj"
