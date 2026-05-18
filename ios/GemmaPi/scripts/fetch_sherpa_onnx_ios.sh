#!/usr/bin/env bash
set -euo pipefail

APP_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
VERSION="${SHERPA_ONNX_VERSION:-1.13.2}"
TARGET_DIR="$APP_DIR/LocalSherpaOnnx"
STAMP="$TARGET_DIR/.sherpa-onnx-ios-$VERSION.ready"
ARCHIVE="$TARGET_DIR/cache/sherpa-onnx-v$VERSION-ios.tar.bz2"
URL="https://github.com/k2-fsa/sherpa-onnx/releases/download/v$VERSION/sherpa-onnx-v$VERSION-ios.tar.bz2"

if [ -f "$STAMP" ] \
  && [ -d "$TARGET_DIR/sherpa-onnx.xcframework" ] \
  && [ -d "$TARGET_DIR/onnxruntime.xcframework" ] \
  && [ -d "$TARGET_DIR/include/sherpa-onnx/c-api" ]; then
  exit 0
fi

mkdir -p "$TARGET_DIR/cache"

if [ ! -f "$ARCHIVE" ]; then
  echo "Downloading sherpa-onnx iOS $VERSION..."
  curl -L --fail --progress-bar -o "$ARCHIVE" "$URL"
fi

TMP_DIR="$(mktemp -d "${TMPDIR:-/tmp}/sherpa-onnx-ios.XXXXXX")"
trap 'rm -rf "$TMP_DIR"' EXIT

tar -xjf "$ARCHIVE" -C "$TMP_DIR"

rm -rf "$TARGET_DIR/sherpa-onnx.xcframework" "$TARGET_DIR/onnxruntime.xcframework" "$TARGET_DIR/include"
cp -R "$TMP_DIR/build-ios/sherpa-onnx.xcframework" "$TARGET_DIR/sherpa-onnx.xcframework"
cp -R "$TMP_DIR/build-ios/ios-onnxruntime/1.17.1/onnxruntime.xcframework" "$TARGET_DIR/onnxruntime.xcframework"
mkdir -p "$TARGET_DIR/include"
cp -R "$TARGET_DIR/sherpa-onnx.xcframework/ios-arm64/Headers/"* "$TARGET_DIR/include/"
touch "$STAMP"
