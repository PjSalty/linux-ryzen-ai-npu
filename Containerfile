# syntax=docker/dockerfile:1
#
# AMD Ryzen AI Software 1.9.0 user space for the XDNA 2 NPU, on any Linux distro.
#
# The container shares nothing with the host except /dev/accel/accel0. It uses
# the host kernel's in-tree amdxdna driver and the firmware the host already
# loads. It installs no kernel module, no DKMS package, no udev rule and no
# firmware, on the host or in the image.
#
# AMD's packages are not part of this repository. scripts/fetch-amd.sh downloads
# them from download.amd.com after you accept the AMD EULA, and the build reads
# them through a read-only named build context (scripts/build.sh runs this):
#
#   podman build --build-context amd=./downloads --build-arg AMD_EULA=accept -t ryzen-npu:1.9.0 .
#
# Versions (checked 2026-10-31):
#   ubuntu:24.04 digest from docker.io, built 2026-10-25 (24.04.5 LTS)
#   XRT 2.26.89 and the amdxdna plugin 2.26.260102.59 (RAI_1.9_Linux_NPU_XRT.zip)
#   Ryzen AI Software 1.9.0 (ryzen_ai-1.9.0.tgz)
#
# No USER: rootless podman maps the container's root to the user who runs it,
# and scripts/run.sh passes --user for rootful docker. Ubuntu package versions
# are not pinned because security updates replace them; the base digest is.

FROM docker.io/library/ubuntu:24.04@sha256:78d4744ead1f5b19b04d0df558ab182a5300903900755e515b2ca4d10eb2044d

# Lets scripts/uninstall.sh remove this build's layers and nothing else.
LABEL ryzen-npu.build="1"

SHELL ["/bin/bash", "-o", "pipefail", "-c"]

ARG AMD_EULA=
RUN if [[ "${AMD_EULA}" != "accept" ]]; then \
      echo "Read the AMD EULA at https://ryzenai.docs.amd.com/en/latest/licenses.html"; \
      echo "and pass --build-arg AMD_EULA=accept if you agree to it."; \
      exit 1; \
    fi

# libgomp1, libncurses6 and libpython3.12t64 are libraries the Ryzen AI runtime
# links against that a desktop Ubuntu has and a minimal image does not. Without
# libncurses6 the Vitis AI EP fails to load and ONNX Runtime silently falls
# back to the CPU. The asm link is the one AMD's installer asks for. libatomic1
# is for Mitsuba's drjit (npubench.testscene).
ENV DEBIAN_FRONTEND=noninteractive
RUN apt-get update \
 && apt-get install -y --no-install-recommends \
      ca-certificates file libboost-filesystem1.83.0 libboost-program-options1.83.0 \
      libatomic1 libgomp1 libncurses6 libpython3.12t64 libssl3t64 libuuid1 linux-libc-dev \
      ocl-icd-libopencl1 patch pciutils python3 python3.12 python3.12-venv unzip zip \
 && rm -rf /var/lib/apt/lists/* \
 && ln -s /usr/include/asm-generic /usr/include/asm

# XRT user space. The plugin package also carries a DKMS kernel module,
# firmware and a postinst that runs rmmod and modprobe; none of that is wanted,
# so only its /opt/xilinx/xrt files are unpacked.
WORKDIR /tmp/xrt
RUN --mount=type=bind,from=amd,target=/amd \
    set -eux; \
    unzip -q /amd/RAI_1.9_Linux_NPU_XRT.zip; \
    apt-get update; \
    apt-get install -y --no-install-recommends \
      ./xrt_202620.2.26.89_24.04-amd64-base.deb \
      ./xrt_202620.2.26.89_24.04-amd64-npu.deb; \
    dpkg-deb -x xrt_plugin.2.26.260102.59.release_24.04-amd64-amdxdna.deb plugin; \
    cp -a plugin/opt/xilinx/xrt/. /opt/xilinx/xrt/; \
    rm -rf /tmp/xrt/* /var/lib/apt/lists/*

# Ryzen AI Software, installed by AMD's own script, unmodified; the script has
# to run from the folder it was unpacked into. AMD's archives store files owned
# by uid 95777, which a rootless build cannot map, so every tar (including the
# ones the script runs) keeps the extracting user instead.
WORKDIR /tmp/rai
RUN --mount=type=bind,from=amd,target=/amd \
    set -eux; \
    export TAR_OPTIONS=--no-same-owner; \
    tar -xzf /amd/ryzen_ai-1.9.0.tgz; \
    bash ./install_ryzen_ai.sh -a yes -p /opt/ryzen-ai/venv; \
    rm -rf /tmp/rai/* /root/.cache

COPY scripts/entrypoint.sh /usr/local/bin/entrypoint.sh
WORKDIR /work
ENTRYPOINT ["/usr/local/bin/entrypoint.sh"]
CMD ["bash"]
