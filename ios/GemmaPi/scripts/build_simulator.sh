#!/usr/bin/env bash
set -euo pipefail

APP_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

"$APP_DIR/scripts/generate_project.sh"

xcodebuild \
  -project "$APP_DIR/GemmaPi.xcodeproj" \
  -scheme GemmaPi \
  -destination 'platform=iOS Simulator,name=iPhone 17 Pro,OS=latest' \
  build
