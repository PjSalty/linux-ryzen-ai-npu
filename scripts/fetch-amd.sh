#!/usr/bin/env bash
#
# Script: fetch-amd.sh
# Description: Downloads the AMD Ryzen AI Software 1.9.0 Linux packages from
#   download.amd.com and checks their SHA-256. The files are AMD software
#   under the AMD EULA: https://ryzenai.docs.amd.com/en/latest/licenses.html
# Usage: scripts/fetch-amd.sh --accept-amd-eula [dest-dir]
#

set -euo pipefail

readonly BASE_URL="https://download.amd.com/opendownload/RyzenAI/1.9.0"
# SHA-256 of the files as published on 2026-10-30 (checked 2026-10-31).
readonly -a FILES=(RAI_1.9_Linux_NPU_XRT.zip ryzen_ai-1.9.0.tgz)
readonly -a SUMS=(
  074c981fad839ea4b7623337f63ed772e3469f5212d16c220b0f6f8b002b7877
  f42461961d6b4f395704ed343fef3cd6528eec285101fa52ea02dcd904f0d07f
)

partial=""

usage() {
  cat <<EOF
Usage: $(basename "$0") --accept-amd-eula [dest-dir]

Downloads RAI_1.9_Linux_NPU_XRT.zip (90 MB) and ryzen_ai-1.9.0.tgz (12.9 GB)
into dest-dir (default: downloads) and verifies their SHA-256.
Read the AMD EULA first: https://ryzenai.docs.amd.com/en/latest/licenses.html
EOF
}

log() { echo "[fetch-amd] $*" >&2; }

cleanup() {
  if [[ -n "${partial}" && -e "${partial}" ]]; then
    rm -f -- "${partial}"
  fi
}
trap cleanup EXIT

main() {
  if [[ "${1:-}" != "--accept-amd-eula" ]]; then
    usage >&2
    exit 2
  fi
  local dest="${2:-downloads}"
  mkdir -p -- "${dest}"

  local i f
  for i in "${!FILES[@]}"; do
    f="${FILES[${i}]}"
    if [[ ! -s "${dest}/${f}" ]]; then
      log "downloading ${f}"
      partial="${dest}/${f}.part"
      curl -fL --retry 3 -o "${partial}" "${BASE_URL}/${f}"
      mv -- "${partial}" "${dest}/${f}"
      partial=""
    fi
    log "verifying ${f}"
    echo "${SUMS[${i}]}  ${dest}/${f}" | sha256sum -c -
  done
}

main "$@"
