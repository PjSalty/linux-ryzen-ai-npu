#!/usr/bin/env bash
#
# Script: uninstall.sh
# Description: Removes everything this repository created outside its own
#   folder: the container image and the build layers it left behind. The host
#   kernel, driver, firmware and packages are never changed, so nothing else
#   needs undoing. Delete the repository folder (with its downloads/, models/
#   and cache/ folders) yourself afterwards.
# Usage: scripts/uninstall.sh
#

set -euo pipefail

readonly ENGINE="${CONTAINER_ENGINE:-podman}"
readonly IMAGE="${NPU_IMAGE:-localhost/ryzen-npu:1.9.0}"

usage() {
  echo "Usage: $(basename "$0")   (removes ${IMAGE} and this repo's build layers)"
}

main() {
  if [[ "${1:-}" == "-h" || "${1:-}" == "--help" ]]; then
    usage
    exit 0
  fi
  if "${ENGINE}" image inspect "${IMAGE}" >/dev/null 2>&1; then
    echo "[uninstall] removing ${IMAGE}" >&2
    "${ENGINE}" image rm "${IMAGE}"
  fi
  # Untagged layers left by this repository's builds only (they carry its label).
  "${ENGINE}" image prune -f --filter label=ryzen-npu.build=1
  echo "[uninstall] done; the host kernel, driver, firmware and packages were never touched." >&2
}

main "$@"
