#!/usr/bin/env bash
#
# Script: run.sh
# Description: Runs a command in the ryzen-npu container with only the NPU
#   passed through. Memory and swap are capped (an out-of-memory kill stays
#   inside the container), CPU time is capped and the scheduling weight is the
#   lowest, so the container cannot starve the rest of the machine.
#   Environment: NPU_MEMORY (14g), NPU_CPUS (6), NPU_IMAGE, CONTAINER_ENGINE
#   (podman or docker).
# Usage: scripts/run.sh <command> [args...]
#

set -euo pipefail

readonly ENGINE="${CONTAINER_ENGINE:-podman}"
readonly IMAGE="${NPU_IMAGE:-localhost/ryzen-npu:1.9.0}"
readonly MEMORY="${NPU_MEMORY:-14g}"
readonly CPUS="${NPU_CPUS:-6}"

usage() {
  cat <<EOF
Usage: $(basename "$0") <command> [args...]

Example: $(basename "$0") python -m npubench.bench --model m.onnx --device npu
EOF
}

main() {
  if [[ $# -eq 0 || "${1}" == "-h" || "${1}" == "--help" ]]; then
    usage >&2
    exit 2
  fi
  if [[ ! -e /dev/accel/accel0 ]]; then
    echo "No /dev/accel/accel0. Is the amdxdna kernel driver loaded (Linux 6.14 or newer)?" >&2
    exit 1
  fi

  local -a args=(
    --device /dev/accel/accel0
    --ulimit memlock=-1:-1
    --memory "${MEMORY}" --memory-swap "${MEMORY}"
    --cpus "${CPUS}" --cpu-shares 2
    --security-opt no-new-privileges
    --cap-drop ALL
    -v "${PWD}:/work:z" -w /work
  )
  if [[ "${ENGINE}" == "docker" ]]; then
    # Rootful docker would write root-owned files into the bind mount.
    args+=(--user "$(id -u):$(id -g)" -e HOME=/tmp)
  fi
  if [[ -t 0 && -t 1 ]]; then
    args+=(-it)
  fi

  exec "${ENGINE}" run "${args[@]}" --rm "${IMAGE}" "$@"
}

main "$@"
