#!/usr/bin/env bash
set -euo pipefail

APP_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
BUNDLE_ID="${IOS_BUNDLE_ID:-com.gemma4robot.gemmapi}"
DEFAULT_DEVELOPMENT_TEAM="${IOS_DEFAULT_DEVELOPMENT_TEAM:-HT74ZXZB82}"
DERIVED_DATA="${IOS_DERIVED_DATA:-$APP_DIR/DerivedData}"
CONFIGURATION="${IOS_CONFIGURATION:-Debug}"
ACTION="${1:-run}"
if [ "$#" -gt 0 ]; then
  shift
fi
LAUNCH_ARGS=("$@")

find_device_id() {
  if [ -n "${IOS_DEVICE_ID:-}" ]; then
    printf '%s\n' "$IOS_DEVICE_ID"
    return
  fi

  xcrun xctrace list devices 2>/dev/null \
    | awk '/iPhone/ && /\([0-9A-F-]{25,}\)/ {print $NF; exit}' \
    | tr -d '()'
}

find_team_id() {
  if [ -n "${IOS_DEVELOPMENT_TEAM:-}" ]; then
    printf '%s\n' "$IOS_DEVELOPMENT_TEAM"
    return
  fi

  if [ -n "$DEFAULT_DEVELOPMENT_TEAM" ]; then
    printf '%s\n' "$DEFAULT_DEVELOPMENT_TEAM"
    return
  fi

  profile_dirs=(
    "$HOME/Library/Developer/Xcode/UserData/Provisioning Profiles"
    "$HOME/Library/MobileDevice/Provisioning Profiles"
  )

  for profile_dir in "${profile_dirs[@]}"; do
    [ -d "$profile_dir" ] || continue
    while IFS= read -r -d '' profile; do
      app_id="$(
        security cms -D -i "$profile" 2>/dev/null \
          | plutil -extract Entitlements.application-identifier raw -o - - 2>/dev/null \
          || true
      )"
      if [[ "$app_id" == *."$BUNDLE_ID" ]]; then
        security cms -D -i "$profile" 2>/dev/null \
          | plutil -extract TeamIdentifier.0 raw -o - - 2>/dev/null
        return 0
      fi
    done < <(find "$profile_dir" -maxdepth 1 -type f -name '*.mobileprovision' -print0)
  done

  return 1
}

DEVICE_ID="$(find_device_id)"
if [ -z "$DEVICE_ID" ]; then
  echo "No available iPhone found. Set IOS_DEVICE_ID explicitly." >&2
  exit 1
fi

TEAM_ID="$(find_team_id)"
if [ -z "$TEAM_ID" ]; then
  echo "No Apple Development signing team found. Set IOS_DEVELOPMENT_TEAM." >&2
  exit 1
fi

APP_PATH="$DERIVED_DATA/Build/Products/${CONFIGURATION}-iphoneos/GemmaPi.app"

build_app() {
  "$APP_DIR/scripts/generate_project.sh"
  if [ -f "$APP_DIR/Podfile" ]; then
    (cd "$APP_DIR" && pod install)
  fi
  build_container_args=(-project "$APP_DIR/GemmaPi.xcodeproj")
  if [ -d "$APP_DIR/GemmaPi.xcworkspace" ]; then
    build_container_args=(-workspace "$APP_DIR/GemmaPi.xcworkspace")
  fi
  xcodebuild \
    "${build_container_args[@]}" \
    -scheme GemmaPi \
    -configuration "$CONFIGURATION" \
    -destination "platform=iOS,id=$DEVICE_ID" \
    -derivedDataPath "$DERIVED_DATA" \
    PRODUCT_BUNDLE_IDENTIFIER="$BUNDLE_ID" \
    DEVELOPMENT_TEAM="$TEAM_ID" \
    -allowProvisioningUpdates \
    build
}

install_app() {
  if [ ! -d "$APP_PATH" ]; then
    build_app
  fi
  xcrun devicectl device install app --device "$DEVICE_ID" "$APP_PATH"
}

launch_app() {
  if [ "${#LAUNCH_ARGS[@]}" -gt 0 ]; then
    xcrun devicectl device process launch \
      --device "$DEVICE_ID" \
      --terminate-existing \
      "$BUNDLE_ID" \
      "${LAUNCH_ARGS[@]}"
  else
    xcrun devicectl device process launch \
      --device "$DEVICE_ID" \
      --terminate-existing \
      "$BUNDLE_ID"
  fi
}

console_app() {
  if [ "${#LAUNCH_ARGS[@]}" -gt 0 ]; then
    xcrun devicectl device process launch \
      --device "$DEVICE_ID" \
      --terminate-existing \
      --console \
      "$BUNDLE_ID" \
      "${LAUNCH_ARGS[@]}"
  else
    xcrun devicectl device process launch \
      --device "$DEVICE_ID" \
      --terminate-existing \
      --console \
      "$BUNDLE_ID"
  fi
}

list_crashes() {
  xcrun devicectl device info files \
    --device "$DEVICE_ID" \
    --domain-type systemCrashLogs \
    --subdirectory . \
    --filter "Name CONTAINS 'GemmaPi'" \
    --sort-by "Modification date" \
    --columns '*'
}

case "$ACTION" in
  build)
    build_app
    ;;
  install)
    build_app
    install_app
    ;;
  launch)
    launch_app
    ;;
  console|logs)
    console_app
    ;;
  run)
    build_app
    install_app
    console_app
    ;;
  crashes)
    list_crashes
    ;;
  *)
    cat >&2 <<EOF
Usage: $0 [build|install|launch|console|logs|run|crashes] [app launch args...]

Environment:
  IOS_DEVICE_ID          iPhone UDID/CoreDevice identifier. Default: first paired iPhone from xctrace.
  IOS_DEVELOPMENT_TEAM   Apple development team id. Default: $DEFAULT_DEVELOPMENT_TEAM, or matching local provisioning profile if IOS_DEFAULT_DEVELOPMENT_TEAM is empty.
  IOS_DERIVED_DATA       DerivedData directory. Default: ios/GemmaPi/DerivedData.
  IOS_CONFIGURATION      Xcode configuration. Default: Debug.
EOF
    exit 2
    ;;
esac
