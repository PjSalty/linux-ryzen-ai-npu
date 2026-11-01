#!/usr/bin/env bash
#
# Script: build.sh
# Description: Builds the ryzen-npu image from the AMD files in ./downloads
#   (see fetch-amd.sh). Extra flags for the build engine go in NPU_BUILD_OPTS,
#   for example to build at the lowest CPU weight with a memory cap, on CPUs
#   16-23 only (rootless podman usually has no cpuset controller, so pin with
#   taskset instead):
#     NPU_BUILD_OPTS="--cpu-shares 2 --memory 10g" \
#       taskset -c 16-23 scripts/build.sh --accept-amd-eula
# Usage: scripts/build.sh --accept-amd-eula
#

set -euo pipefail

readonly ENGINE="${CONTAINER_ENGINE:-podman}"
readonly IMAGE="${NPU_IMAGE:-localhost/ryzen-npu:1.9.0}"
readonly -a NEEDED=(RAI_1.9_Linux_NPU_XRT.zip ryzen_ai-1.9.0.tgz)

usage() {
  cat <<EOF
Usage: $(basename "$0") --accept-amd-eula

Builds ${IMAGE} with ${ENGINE}. The image contains AMD Ryzen AI Software
under the AMD EULA: https://ryzenai.docs.amd.com/en/latest/licenses.html
EOF
}

main() {
  if [[ "${1:-}" != "--accept-amd-eula" ]]; then
    usage >&2
    exit 2
  fi
  cd "$(dirname "$0")/.." || exit 1

  local f
  for f in "${NEEDED[@]}"; do
    if [[ ! -s "downloads/${f}" ]]; then
      echo "downloads/${f} is missing: run scripts/fetch-amd.sh first" >&2
      exit 1
    fi
  done

  local -a extra=()
  read -r -a extra <<<"${NPU_BUILD_OPTS:-}"
  echo "[build] ${ENGINE} build -t ${IMAGE}" >&2
  exec "${ENGINE}" build --build-context amd=./downloads --build-arg AMD_EULA=accept \
    "${extra[@]}" -t "${IMAGE}" -f Containerfile .
}

main "$@"
