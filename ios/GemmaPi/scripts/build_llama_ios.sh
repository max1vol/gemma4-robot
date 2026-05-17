#!/usr/bin/env bash
set -euo pipefail

APP_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
LOCAL_DIR="$APP_DIR/LocalLlama"
SRC_DIR="$LOCAL_DIR/llama.cpp"
BUILD_DIR="$LOCAL_DIR/build-ios-device"
FRAMEWORK_DIR="$LOCAL_DIR/llama.framework"
TEMP_DIR="$LOCAL_DIR/temp"
IOS_MIN_OS_VERSION="${IOS_MIN_OS_VERSION:-17.0}"
LLAMA_REPO="${LLAMA_REPO:-https://github.com/ggml-org/llama.cpp.git}"
LLAMA_REF="${LLAMA_REF:-master}"

check_tool() {
  if ! command -v "$1" >/dev/null 2>&1; then
    echo "Missing required tool: $1" >&2
    exit 1
  fi
}

check_tool git
check_tool cmake
check_tool xcrun

mkdir -p "$LOCAL_DIR"

if [ ! -d "$SRC_DIR/.git" ]; then
  git clone --depth 1 --branch "$LLAMA_REF" "$LLAMA_REPO" "$SRC_DIR"
else
  current_remote="$(git -C "$SRC_DIR" remote get-url origin)"
  if [ "$current_remote" != "$LLAMA_REPO" ]; then
    git -C "$SRC_DIR" remote set-url origin "$LLAMA_REPO"
  fi
  git -C "$SRC_DIR" fetch --depth 1 origin "$LLAMA_REF"
  git -C "$SRC_DIR" checkout FETCH_HEAD
fi

cd "$SRC_DIR"

COMMON_C_FLAGS="-Wno-macro-redefined -Wno-shorten-64-to-32 -Wno-unused-command-line-argument -g"
COMMON_CXX_FLAGS="-Wno-macro-redefined -Wno-shorten-64-to-32 -Wno-unused-command-line-argument -g"

cmake -B "$BUILD_DIR" -G Xcode \
  -DCMAKE_XCODE_ATTRIBUTE_CODE_SIGNING_REQUIRED=NO \
  -DCMAKE_XCODE_ATTRIBUTE_CODE_SIGN_IDENTITY="" \
  -DCMAKE_XCODE_ATTRIBUTE_CODE_SIGNING_ALLOWED=NO \
  -DCMAKE_XCODE_ATTRIBUTE_DEBUG_INFORMATION_FORMAT="dwarf-with-dsym" \
  -DCMAKE_XCODE_ATTRIBUTE_GCC_GENERATE_DEBUGGING_SYMBOLS=YES \
  -DCMAKE_XCODE_ATTRIBUTE_COPY_PHASE_STRIP=NO \
  -DCMAKE_XCODE_ATTRIBUTE_STRIP_INSTALLED_PRODUCT=NO \
  -DBUILD_SHARED_LIBS=OFF \
  -DLLAMA_BUILD_EXAMPLES=OFF \
  -DLLAMA_BUILD_TOOLS=OFF \
  -DLLAMA_BUILD_MTMD=ON \
  -DLLAMA_BUILD_TESTS=OFF \
  -DLLAMA_BUILD_SERVER=OFF \
  -DGGML_METAL=ON \
  -DGGML_METAL_EMBED_LIBRARY=ON \
  -DGGML_METAL_USE_BF16=ON \
  -DGGML_BLAS_DEFAULT=ON \
  -DGGML_NATIVE=OFF \
  -DGGML_OPENMP=OFF \
  -DCMAKE_OSX_DEPLOYMENT_TARGET="$IOS_MIN_OS_VERSION" \
  -DCMAKE_SYSTEM_NAME=iOS \
  -DCMAKE_OSX_SYSROOT=iphoneos \
  -DCMAKE_OSX_ARCHITECTURES=arm64 \
  -DCMAKE_XCODE_ATTRIBUTE_SUPPORTED_PLATFORMS=iphoneos \
  -DCMAKE_C_FLAGS="$COMMON_C_FLAGS" \
  -DCMAKE_CXX_FLAGS="$COMMON_CXX_FLAGS" \
  -DLLAMA_OPENSSL=OFF \
  -S .

cmake --build "$BUILD_DIR" --config Release --target mtmd -- -quiet

rm -rf "$FRAMEWORK_DIR" "$TEMP_DIR"
mkdir -p "$FRAMEWORK_DIR/Headers" "$FRAMEWORK_DIR/Modules" "$TEMP_DIR"

cp include/llama.h "$FRAMEWORK_DIR/Headers/"
cp ggml/include/ggml.h "$FRAMEWORK_DIR/Headers/"
cp ggml/include/ggml-opt.h "$FRAMEWORK_DIR/Headers/"
cp ggml/include/ggml-alloc.h "$FRAMEWORK_DIR/Headers/"
cp ggml/include/ggml-backend.h "$FRAMEWORK_DIR/Headers/"
cp ggml/include/ggml-metal.h "$FRAMEWORK_DIR/Headers/"
cp ggml/include/ggml-cpu.h "$FRAMEWORK_DIR/Headers/"
cp ggml/include/ggml-blas.h "$FRAMEWORK_DIR/Headers/"
cp ggml/include/gguf.h "$FRAMEWORK_DIR/Headers/"
cp tools/mtmd/mtmd.h "$FRAMEWORK_DIR/Headers/"
cp tools/mtmd/mtmd-helper.h "$FRAMEWORK_DIR/Headers/"

cat > "$FRAMEWORK_DIR/Modules/module.modulemap" <<'EOF'
framework module llama {
  header "llama.h"
  header "ggml.h"
  header "ggml-alloc.h"
  header "ggml-backend.h"
  header "ggml-metal.h"
  header "ggml-cpu.h"
  header "ggml-blas.h"
  header "gguf.h"
  header "mtmd.h"
  header "mtmd-helper.h"

  link "c++"
  link framework "Accelerate"
  link framework "Metal"
  link framework "Foundation"

  export *
}
EOF

cat > "$FRAMEWORK_DIR/Info.plist" <<EOF
<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN" "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0">
<dict>
  <key>CFBundleDevelopmentRegion</key>
  <string>en</string>
  <key>CFBundleExecutable</key>
  <string>llama</string>
  <key>CFBundleIdentifier</key>
  <string>org.ggml.llama</string>
  <key>CFBundleInfoDictionaryVersion</key>
  <string>6.0</string>
  <key>CFBundleName</key>
  <string>llama</string>
  <key>CFBundlePackageType</key>
  <string>FMWK</string>
  <key>CFBundleShortVersionString</key>
  <string>1.0</string>
  <key>CFBundleVersion</key>
  <string>1</string>
  <key>MinimumOSVersion</key>
  <string>$IOS_MIN_OS_VERSION</string>
  <key>CFBundleSupportedPlatforms</key>
  <array>
    <string>iPhoneOS</string>
  </array>
  <key>UIDeviceFamily</key>
  <array>
    <integer>1</integer>
    <integer>2</integer>
  </array>
  <key>DTPlatformName</key>
  <string>iphoneos</string>
  <key>DTSDKName</key>
  <string>iphoneos$IOS_MIN_OS_VERSION</string>
</dict>
</plist>
EOF

LIBS=(
  "$BUILD_DIR/src/Release-iphoneos/libllama.a"
  "$BUILD_DIR/ggml/src/Release-iphoneos/libggml.a"
  "$BUILD_DIR/ggml/src/Release-iphoneos/libggml-base.a"
  "$BUILD_DIR/ggml/src/Release-iphoneos/libggml-cpu.a"
  "$BUILD_DIR/ggml/src/ggml-metal/Release-iphoneos/libggml-metal.a"
  "$BUILD_DIR/ggml/src/ggml-blas/Release-iphoneos/libggml-blas.a"
  "$BUILD_DIR/tools/mtmd/Release-iphoneos/libmtmd.a"
)

xcrun libtool -static -o "$TEMP_DIR/combined.a" "${LIBS[@]}" 2>/dev/null

xcrun -sdk iphoneos clang++ -dynamiclib \
  -isysroot "$(xcrun --sdk iphoneos --show-sdk-path)" \
  -arch arm64 \
  -mios-version-min="$IOS_MIN_OS_VERSION" \
  -Wl,-force_load,"$TEMP_DIR/combined.a" \
  -framework Foundation \
  -framework Metal \
  -framework Accelerate \
  -install_name "@rpath/llama.framework/llama" \
  -o "$FRAMEWORK_DIR/llama"

if xcrun -f vtool >/dev/null 2>&1; then
  xcrun vtool -set-build-version ios "$IOS_MIN_OS_VERSION" "$IOS_MIN_OS_VERSION" -replace \
    -output "$FRAMEWORK_DIR/llama" "$FRAMEWORK_DIR/llama"
fi

mkdir -p "$LOCAL_DIR/dSYMs"
xcrun dsymutil "$FRAMEWORK_DIR/llama" -o "$LOCAL_DIR/dSYMs/llama.dSYM"
xcrun strip -S "$FRAMEWORK_DIR/llama" -o "$TEMP_DIR/llama.stripped"
mv "$TEMP_DIR/llama.stripped" "$FRAMEWORK_DIR/llama"
rm -rf "$TEMP_DIR"

codesign --force --sign - "$FRAMEWORK_DIR"
echo "Built $FRAMEWORK_DIR"
