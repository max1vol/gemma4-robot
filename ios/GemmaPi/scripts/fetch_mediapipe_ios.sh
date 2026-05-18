#!/usr/bin/env bash
set -euo pipefail

APP_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
ARTIFACT_DIR="$APP_DIR/LocalMediaPipe/Artifacts"
CACHE_DIR="${TMPDIR:-/tmp}/gemmapi-mediapipe-cache"

MEDIAPIPE_VERSION="0.10.35"
COMMON_URL="https://dl.google.com/cpdc/20260427-204214/MediaPipeTasksCommon-${MEDIAPIPE_VERSION}.tar.gz"
VISION_URL="https://dl.google.com/cpdc/20260427-204221/MediaPipeTasksVision-${MEDIAPIPE_VERSION}.tar.gz"

copy_from_pods_if_available() {
  local name="$1"
  local pod_framework="$APP_DIR/Pods/$name/frameworks/$name.xcframework"
  if [ -d "$ARTIFACT_DIR/$name.xcframework" ]; then
    return 0
  fi
  if [ ! -d "$pod_framework" ]; then
    return 1
  fi

  mkdir -p "$ARTIFACT_DIR"
  cp -R "$pod_framework" "$ARTIFACT_DIR/"
  if [ "$name" = "MediaPipeTasksCommon" ]; then
    rm -rf "$ARTIFACT_DIR/MediaPipeTasksCommonGraph"
    mkdir -p "$ARTIFACT_DIR/MediaPipeTasksCommonGraph"
    cp "$APP_DIR"/Pods/MediaPipeTasksCommon/frameworks/graph_libraries/*.a "$ARTIFACT_DIR/MediaPipeTasksCommonGraph/"
  fi
}

fetch_archive() {
  local name="$1"
  local url="$2"
  local archive="$CACHE_DIR/${name}-${MEDIAPIPE_VERSION}.tar.gz"
  local unpack_dir="$CACHE_DIR/${name}-${MEDIAPIPE_VERSION}"

  copy_from_pods_if_available "$name" && return

  mkdir -p "$CACHE_DIR" "$ARTIFACT_DIR"
  if [ ! -f "$archive" ]; then
    curl -fL --retry 3 --retry-delay 2 "$url" -o "$archive"
  fi

  rm -rf "$unpack_dir"
  mkdir -p "$unpack_dir"
  tar -xzf "$archive" -C "$unpack_dir"

  rm -rf "$ARTIFACT_DIR/$name.xcframework"
  cp -R "$unpack_dir/frameworks/$name.xcframework" "$ARTIFACT_DIR/"

  if [ "$name" = "MediaPipeTasksCommon" ]; then
    rm -rf "$ARTIFACT_DIR/MediaPipeTasksCommonGraph"
    mkdir -p "$ARTIFACT_DIR/MediaPipeTasksCommonGraph"
    cp "$unpack_dir"/frameworks/graph_libraries/*.a "$ARTIFACT_DIR/MediaPipeTasksCommonGraph/"
  fi
}

fetch_archive "MediaPipeTasksCommon" "$COMMON_URL"
fetch_archive "MediaPipeTasksVision" "$VISION_URL"
