#!/bin/sh
set -eu

ROOT_DIR="$(cd "$(dirname "$0")/.." && pwd)"
OUT_DIR="${ROOT_DIR}/LocalLiteRT/ios_arm64"
TMP_DIR="$(mktemp -d)"
trap 'rm -rf "${TMP_DIR}"' EXIT

LITERT_LM_TAG="${LITERT_LM_TAG:-v0.11.0}"
LITERT_LM_PREBUILT_BASE="${LITERT_LM_PREBUILT_BASE:-https://github.com/google-ai-edge/LiteRT-LM/raw/${LITERT_LM_TAG}/prebuilt/ios_arm64}"
ZIP_URL="${LITERT_PREBUILTS_URL:-}"
ZIP_PATH="${TMP_DIR}/litert_prebuilts.zip"

download_dylib() {
  name="$1"
  url="${LITERT_LM_PREBUILT_BASE}/${name}"
  out="${TMP_DIR}/${name}"
  echo "Downloading ${name} from ${url}"
  curl -L --fail --show-error --output "${out}" "${url}"

  if ! file "${out}" | grep -q "Mach-O 64-bit dynamically linked shared library arm64"; then
    echo "error: ${out} is not an iOS arm64 Mach-O dylib" >&2
    file "${out}" >&2
    exit 1
  fi

  cp -f "${out}" "${OUT_DIR}/${name}"
}

mkdir -p "${OUT_DIR}"

if [ -n "${ZIP_URL}" ]; then
  echo "Downloading LiteRT prebuilts from ${ZIP_URL}"
  curl -L --fail --show-error --output "${ZIP_PATH}" "${ZIP_URL}"

  unzip -q "${ZIP_PATH}" -d "${TMP_DIR}/prebuilts"

  for dylib_name in libLiteRtMetalAccelerator.dylib; do
    dylib_path="${TMP_DIR}/prebuilts/ios_arm64/${dylib_name}"
    if [ -f "${dylib_path}" ]; then
      if ! file "${dylib_path}" | grep -q "Mach-O 64-bit dynamically linked shared library arm64"; then
        echo "error: ${dylib_path} is not an iOS arm64 Mach-O dylib" >&2
        file "${dylib_path}" >&2
        exit 1
      fi
      cp -f "${dylib_path}" "${OUT_DIR}/${dylib_name}"
    fi
  done
  for dylib_name in libLiteRtMetalAccelerator.dylib; do
    if [ ! -f "${OUT_DIR}/${dylib_name}" ]; then
      download_dylib "${dylib_name}"
    fi
  done
else
  for dylib_name in libLiteRtMetalAccelerator.dylib; do
    download_dylib "${dylib_name}"
  done
fi

if [ "${FETCH_LITERT_TOPK_METAL:-0}" = "1" ]; then
  download_dylib "libLiteRtTopKMetalSampler.dylib"
fi

echo "Wrote LiteRT GPU dylibs to ${OUT_DIR}"
find "${OUT_DIR}" -maxdepth 1 -type f -name '*.dylib' -print
