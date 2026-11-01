#!/usr/bin/env bash
# Sets up the Ryzen AI venv and XRT, then runs the given command.
set -euo pipefail

# AMD's activate script and XRT's setup script reference unset variables.
set +u
# shellcheck disable=SC1091
source /opt/ryzen-ai/venv/bin/activate
# XRT goes in front of the venv's library folders. voe/lib bundles its own
# libxrt_coreutil, and XRT looks for libxrt_core next to whichever coreutil
# loads first; voe/lib has none, so with voe/lib first the NPU runner fails
# with "No such library .../voe/lib/libxrt_core.so.2". AMD's .deb installer
# uses the same order ("XRT stays at the front, Ryzen AI appended").
# shellcheck disable=SC1091
source /opt/xilinx/xrt/setup.sh >/dev/null
set -u
# From AMD's Linux installation guide (Ryzen AI 1.9.0).
export LD_LIBRARY_PATH="/lib/x86_64-linux-gnu:${RYZEN_AI_INSTALLATION_PATH}/onnxruntime/lib/:${LD_LIBRARY_PATH}"

exec "$@"
