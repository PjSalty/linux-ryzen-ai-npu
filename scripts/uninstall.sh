#!/usr/bin/env bash
#
# Script: uninstall.sh
# Description: Removes everything this repository created outside its own
#   folder: the NPU image, the iGPU baseline image (Containerfile.rocm) and the
#   build layers they left behind. The host kernel, driver, firmware and
#   packages are never changed, so nothing else needs undoing. Delete the
#   repository folder (with its downloads/, models/ and cache/ folders)
#   yourself afterwards.
#   Environment: NPU_IMAGE, NPU_ROCM_IMAGE, CONTAINER_ENGINE (podman or docker).
# Usage: scripts/uninstall.sh
#

set -euo pipefail

readonly ENGINE="${CONTAINER_ENGINE:-podman}"
readonly IMAGE="${NPU_IMAGE:-localhost/ryzen-npu:1.9.0}"
readonly ROCM_IMAGE="${NPU_ROCM_IMAGE:-localhost/ryzen-npu-rocm:7.2.4}"

usage() {
  echo "Usage: $(basename "$0")   (removes ${IMAGE}, ${ROCM_IMAGE} and this repo's build layers)"
}

main() {
  if [[ "${1:-}" == "-h" || "${1:-}" == "--help" ]]; then
    usage
    exit 0
  fi
  local img
  for img in "${IMAGE}" "${ROCM_IMAGE}"; do
    # Either image may never have been built.
    if "${ENGINE}" image inspect "${img}" >/dev/null 2>&1; then
      echo "[uninstall] removing ${img}" >&2
      "${ENGINE}" image rm "${img}"
    fi
  done
  # Untagged layers left by this repository's builds only (they carry its label).
  "${ENGINE}" image prune -f --filter label=ryzen-npu.build=1
  echo "[uninstall] done; the host kernel, driver, firmware and packages were never touched." >&2
}

main "$@"
