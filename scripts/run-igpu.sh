#!/usr/bin/env bash
#
# Script: run-igpu.sh
# Description: Runs a command in the ROCm baseline container (Containerfile.rocm)
#   with only the iGPU compute nodes passed through (/dev/kfd and the render
#   node), with the same memory, CPU and privilege limits as run.sh.
#   Environment: NPU_MEMORY (8g), NPU_CPUS (4), NPU_RENDER_NODE
#   (/dev/dri/renderD128), NPU_ROCM_IMAGE, CONTAINER_ENGINE (podman or docker).
# Usage: scripts/run-igpu.sh <command> [args...]
#

set -euo pipefail

readonly ENGINE="${CONTAINER_ENGINE:-podman}"
readonly IMAGE="${NPU_ROCM_IMAGE:-localhost/ryzen-npu-rocm:7.2.4}"
readonly MEMORY="${NPU_MEMORY:-8g}"
readonly CPUS="${NPU_CPUS:-4}"
readonly RENDER_NODE="${NPU_RENDER_NODE:-/dev/dri/renderD128}"

usage() {
  cat <<EOF
Usage: $(basename "$0") <command> [args...]

Example: $(basename "$0") oidnDenoise -d hip --ld
EOF
}

main() {
  if [[ $# -eq 0 || "${1}" == "-h" || "${1}" == "--help" ]]; then
    usage >&2
    exit 2
  fi
  local dev
  for dev in /dev/kfd "${RENDER_NODE}"; do
    if [[ ! -e "${dev}" ]]; then
      echo "No ${dev}. Is the amdgpu kernel driver loaded?" >&2
      exit 1
    fi
  done

  local -a args=(
    --device /dev/kfd
    --device "${RENDER_NODE}"
    --memory "${MEMORY}" --memory-swap "${MEMORY}"
    --cpus "${CPUS}" --cpu-shares 2
    --security-opt no-new-privileges
    --cap-drop ALL
    -v "${PWD}:/work:z" -w /work
  )
  if [[ "${ENGINE}" == "docker" ]]; then
    args+=(--user "$(id -u):$(id -g)" --group-add video --group-add render -e HOME=/tmp)
  fi
  if [[ -t 0 && -t 1 ]]; then
    args+=(-it)
  fi

  exec "${ENGINE}" run "${args[@]}" --rm "${IMAGE}" "$@"
}

main "$@"
