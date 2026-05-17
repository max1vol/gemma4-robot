#!/bin/sh
set -eu

APP_FRAMEWORKS_DIR="${TARGET_BUILD_DIR}/${FRAMEWORKS_FOLDER_PATH}"
APP_FRAMEWORK="${APP_FRAMEWORKS_DIR}/CLiteRTLM.framework"
LOCAL_LITERT_DIR="${SRCROOT}/LocalLiteRT/ios_arm64"

can_sign() {
  [ "${CODE_SIGNING_ALLOWED:-YES}" != "NO" ] && [ -n "${EXPANDED_CODE_SIGN_IDENTITY:-}" ]
}

sign_file() {
  if can_sign; then
    /usr/bin/codesign --force --sign "${EXPANDED_CODE_SIGN_IDENTITY}" --timestamp=none "$1"
  fi
}

mkdir -p "${APP_FRAMEWORKS_DIR}"

NESTED_CONSTRAINT_DYLIB="${APP_FRAMEWORK}/libGemmaModelConstraintProvider.dylib"
PRODUCT_CONSTRAINT_DYLIB="${BUILT_PRODUCTS_DIR}/CLiteRTLM.framework/libGemmaModelConstraintProvider.dylib"
ROOT_CONSTRAINT_DYLIB="${APP_FRAMEWORKS_DIR}/libGemmaModelConstraintProvider.dylib"

CONSTRAINT_SOURCE=""
if [ -f "${NESTED_CONSTRAINT_DYLIB}" ]; then
  CONSTRAINT_SOURCE="${NESTED_CONSTRAINT_DYLIB}"
elif [ -f "${PRODUCT_CONSTRAINT_DYLIB}" ]; then
  CONSTRAINT_SOURCE="${PRODUCT_CONSTRAINT_DYLIB}"
fi

if [ -n "${CONSTRAINT_SOURCE}" ]; then
  cp -f "${CONSTRAINT_SOURCE}" "${ROOT_CONSTRAINT_DYLIB}"
  sign_file "${ROOT_CONSTRAINT_DYLIB}"
  if [ -f "${NESTED_CONSTRAINT_DYLIB}" ]; then
    sign_file "${NESTED_CONSTRAINT_DYLIB}"
  fi
else
  echo "warning: LiteRT-LM constraint provider dylib was not found; app launch may fail if CLiteRTLM requires it"
fi

for stale_dylib_name in libLiteRtMetalAccelerator.dylib libLiteRtTopKMetalSampler.dylib; do
  rm -f "${APP_FRAMEWORKS_DIR}/${stale_dylib_name}"
  rm -f "${APP_FRAMEWORK}/${stale_dylib_name}"
done

METAL_DYLIB="${LOCAL_LITERT_DIR}/libLiteRtMetalAccelerator.dylib"
if [ -f "${METAL_DYLIB}" ]; then
  ROOT_METAL_DYLIB="${APP_FRAMEWORKS_DIR}/libLiteRtMetalAccelerator.dylib"
  cp -f "${METAL_DYLIB}" "${ROOT_METAL_DYLIB}"
  sign_file "${ROOT_METAL_DYLIB}"
else
  echo "warning: ${METAL_DYLIB} not found; run scripts/fetch_litert_gpu_prebuilts.sh before building for GPU support"
fi

if can_sign && [ -d "${APP_FRAMEWORK}" ]; then
  /usr/bin/codesign --force --sign "${EXPANDED_CODE_SIGN_IDENTITY}" --timestamp=none "${APP_FRAMEWORK}"
elif ! can_sign; then
  echo "Skipping LiteRT-LM dylib signing because code signing is disabled or no identity is available"
fi
