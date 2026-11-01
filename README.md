# linux-ryzen-ai-npu

Run ONNX models on the XDNA 2 NPU of an AMD Ryzen AI laptop from Linux, and
measure them against the CPU and the iGPU of the same chip, with the same model
and the same input.

The NPU runs AMD's own Ryzen AI Software 1.9.0 inside a rootless container. The
container uses the kernel's in-tree `amdxdna` driver (Linux 6.14 and newer) and
the NPU firmware your distro already ships, and it sees only
`/dev/accel/accel0`. Nothing is installed on the host: no kernel module, no
DKMS package, no udev rule, no firmware. AMD supports its Linux release on
Ubuntu 24.04; the image carries an Ubuntu 24.04 user space, so it also runs on
other distros. It was tested on Arch Linux.

## Results at a glance

| Workload | Runs on the NPU? | Best device here |
|----------|------------------|------------------|
| SigLIP 2 image embeddings (base, 224 px) | Yes: the whole graph, in BF16 | NPU for energy (0.23 J per image), iGPU for speed (6.8 ms) |
| Depth Anything 3 Small (504 px) | Yes: the whole graph, in BF16 | NPU for energy (0.9 J per image), iGPU for speed (44 ms) |
| Open Image Denoise (1920x1280 HDR frame) | Not usefully: BF16 does not compile in practical time, INT8 is too inaccurate | iGPU, with OIDN's own HIP backend (82 ms per frame) |

The NPU's performance power mode, which needs root to set, makes SigLIP 2 and
DA3 Small about 3 times faster on the NPU for about the same energy per image
(see [NPU power mode](#npu-power-mode)).

Details, method and caveats are below. Every NPU speed and power number in this
README comes from a run where the Vitis AI execution provider loaded and the
NPU's own busy counter moved (see [Proof that the NPU ran](#proof-that-the-npu-ran)).
The accuracy runs load the same compiled models from the cache.

## Requirements

- A Ryzen AI processor with an XDNA 2 NPU. AMD lists Linux support from Strix
  Point on; this was tested on a Ryzen AI 9 HX 375.
- Linux 6.14 or newer with the `amdxdna` driver loaded, so that
  `/dev/accel/accel0` exists, and the NPU firmware from `linux-firmware` in
  `/lib/firmware/amdnpu/`.
- Podman (rootless) or Docker. Tested with podman 6.1.2.
- About 50 GB of free disk: 13 GB of AMD downloads and a 28 GB image, whose
  build needs more room while it runs. The optional iGPU baseline image adds
  14 GB.
- For the iGPU baselines only: `/dev/kfd` and a render node from the `amdgpu`
  driver.

`scripts/run.sh` asks for an unlimited locked-memory limit. A rootless
container cannot raise it above your own hard limit; podman then keeps your
limit without an error (8 MiB on a default systemd system). The test machine
had `* hard memlock unlimited` in `/etc/security/limits.conf`. Whether the NPU
runtime works with a lower limit was not tested.

## Get AMD's software

This repository contains no AMD software. The image is built from two files
that AMD publishes for Ryzen AI Software 1.9.0 on Linux, under the
[AMD End User License Agreement](https://ryzenai.docs.amd.com/en/latest/licenses.html):

| File | Size | Contents |
|------|------|----------|
| `RAI_1.9_Linux_NPU_XRT.zip` | 90 MB | XRT 2.26.89 and the amdxdna plugin 2.26.260102.59. Only their user-space files go into the image. |
| `ryzen_ai-1.9.0.tgz` | 12.9 GB | Ryzen AI Software: ONNX Runtime with the Vitis AI execution provider, and the NPU compilers |

Read the EULA, then either let the script download both files from
`download.amd.com` and check their SHA-256:

```bash
scripts/fetch-amd.sh --accept-amd-eula
```

or download them yourself from AMD's
[Linux installation page](https://ryzenai.docs.amd.com/en/latest/linux.html)
(the "Ryzen AI Linux Drivers" and "Ryzen AI Software Installer" links) into
`downloads/`, and run the script anyway: it skips files that are already there
and only verifies them. The image you build contains AMD's software, so AMD's
license applies to it too.

## Build and check

From the repository folder:

```bash
scripts/build.sh --accept-amd-eula     # builds localhost/ryzen-npu:1.9.0
scripts/run.sh bash -c 'cd /opt/ryzen-ai/venv/quicktest && python quicktest.py'
scripts/run.sh python -m npubench.bench --model /opt/ryzen-ai/venv/quicktest/test_model.onnx \
    --device npu --config default --cache-dir cache/npu --seconds 10
```

On the test machine, the checksum check and a clean build (`--no-cache`, on 8
CPUs) took 11 minutes together; writing the 28 GB image layer was more than
half of it. The second command is AMD's own quicktest. Do not trust it alone: while
this image was being made, it reported success with the Vitis AI execution
provider failing to load (the image lacked `libncurses6` at the time) and
everything running on the CPU. The third command runs the same INT8 test model
with this repository's guard and prints its latency, power and the NPU busy
counter; on the test machine it took 1.5 to 1.6 ms per run, with 398 of its
400 nodes on the NPU and the NPU 86 to 87% busy.

To remove everything this repository created outside its folder (the NPU
image, the iGPU baseline image if you built it, and their build layers), run
`scripts/uninstall.sh`. Delete the folder yourself.

## Run your own model

The NPU compiler needs static input shapes. Then:

```bash
# Float32 models: the VAIML compiler runs them in BF16 (npubench/vitisai_config_bf16.json).
scripts/run.sh python -m npubench.bench --model my_model.onnx --device npu --cache-dir cache/npu

# INT8 QDQ models (for example from npubench.quantize): AMD's default INT8 config.
scripts/run.sh python -m npubench.bench --model my_model_int8.onnx --device npu --config default --cache-dir cache/npu
```

The first session on a new model compiles it and stores the result in
`cache/npu/<model name>/`; later sessions load from there in seconds. A BF16
compile can take a long time (see [Known limits](#known-limits)). During a
BF16 compile, the compiler also writes `original-model-signature.txt` and
`original-info-signature.txt` into the current folder, copies them into the
cache at the end and then removes them. Do not delete them, or sync over the
folder, while a compile runs: the compile then fails at its last step.

For the CPU baseline, use `--device cpu --threads 12` instead of `--device
npu`. For the iGPU, build the ROCm image
(`podman build -f Containerfile.rocm -t ryzen-npu-rocm:7.2.4 .`) and run the
same command with `scripts/run-igpu.sh` and `--device igpu`.

## Benchmarks

### Method

- Machine: Ryzen AI 9 HX 375 (12 cores, 24 threads), Radeon 890M iGPU
  (gfx1150), XDNA 2 NPU with firmware 1.1.2.64, 30 GiB of RAM visible to
  Linux. Arch Linux, kernel 7.2.7. On AC power, platform profile
  `performance`, `amd-pstate-epp` governor `performance`. The NPU power mode
  was the default everywhere except in the power-mode table.
- Software: Ryzen AI Software 1.9.0 (its ONNX Runtime 1.30.0) with XRT
  2.26.89 for the NPU and the CPU rows. ROCm 7.2.4 with MIGraphX 2.15.0 and `onnxruntime-migraphx` 1.23.2 for
  the iGPU. Open Image Denoise 2.5.1 for the OIDN library rows.
- Quiet machine: nothing else ran during the tables below.
- Latency: `npubench.bench` runs 3 warm-up runs, then times runs for 20 s
  (at least 10 runs; 12 s for the OIDN per-channel NPU rows).
  `npubench.oidn_cli_bench` times `oidnDenoise -n` and drops the first 2 runs.
  Tables show the median.
- Power: read from the SMU through amdgpu's `gpu_metrics` (format 3.0) every
  50 ms, without root. "Socket W" is the average socket power over the timed
  runs, skipping the first 0.5 s; "idle" is the 5 s before (3 s for the OIDN
  NPU rows). Energy per run is socket power times mean latency, and "above
  idle" uses socket power minus idle power.
- Limits: NPU and iGPU runs had 4 CPUs; CPU baselines had 24 CPUs and 12
  threads.

### SigLIP 2 base (image embeddings, 224 px)

`google/siglip2-base-patch16-224`, the vision tower to the pooled embedding,
exported to ONNX (FP32).

| Device | Precision | ms per image | Images/s | Socket W (idle) | NPU W | J per image (above idle) |
|--------|-----------|-------------:|---------:|----------------:|------:|-------------------------:|
| CPU, ONNX Runtime, 12 threads | FP32 | 72.2 | 13.6 | 16.8 (3.8) | | 1.24 (0.96) |
| iGPU, ONNX Runtime MIGraphX | FP16 | 6.8 | 149.7 | 46.0 (3.7) | | 0.31 (0.28) |
| NPU, ONNX Runtime Vitis AI EP | BF16 | 55.7 | 18.1 | 4.2 (3.3) | 0.56 | 0.23 (0.05) |

The NPU took all 511 graph nodes. Its first session compiled the model in
6.5 minutes; later sessions load it from the cache in 2.2 s. The iGPU session
took 148 s to start, mostly MIGraphX compiling the graph; `npubench.bench`
does not save MIGraphX's compiled model. Accuracy, on 26 rendered interior
images (not included): the NPU embeddings have a mean cosine similarity of
0.9962 to the CPU's FP32 embeddings (worst 0.967), and 23 of the 26 images
(88%) keep the same nearest neighbour.

### Depth Anything 3 Small (monocular depth, 504 px)

`depth-anything/DA3-SMALL`, exported to ONNX (FP32) with a fixed 504x504 input
by the Depth-Anything-3-Onnx exporter; the output is a 504x504 depth map.

| Device | Precision | ms per image | Images/s | Socket W (idle) | NPU W | J per image (above idle) |
|--------|-----------|-------------:|---------:|----------------:|------:|-------------------------:|
| CPU, ONNX Runtime, 12 threads | FP32 | 238 | 4.0 | 16.8 (6.5) | | 4.24 (2.61) |
| iGPU, ONNX Runtime MIGraphX | FP16 | 44.2 | 22.7 | 39.8 (4.2) | | 1.76 (1.57) |
| NPU, ONNX Runtime Vitis AI EP, idle chip | BF16 | 246 | 4.1 | 3.7 (3.3) | 0.51 | 0.90 (0.08) |
| NPU, same, right after the CPU run | BF16 | 181 | 5.5 | 6.2 (5.3) | 0.72 | 1.12 (0.15) |

The two NPU rows are the same model and settings, 26 minutes apart. The
second ran right after the CPU baseline, while the chip's idle power was still
5.3 W, and the NPU ran faster; see [NPU idle clocking](#known-limits). On an
idle chip, the NPU is about as fast as 12 CPU threads at a fifth of the
energy. The NPU took all 857 graph nodes and was 96% busy. Its first session
compiled the model in 22 minutes; later sessions load it from the cache in
1.3 s. The iGPU session took 24 minutes to start, mostly MIGraphX compiling
the graph.
Accuracy, on the same 26 images: the NPU's depth maps differ from the CPU's
FP32 ones by 3.3% on average (mean absolute relative difference; worst image
4.4%), every pixel is within the usual δ < 1.25 ratio, and the Pearson
correlation is 0.993 on average (worst image 0.913).

### Open Image Denoise (1920x1280 HDR frame)

OIDN 2.5.1's `RT` filter at high quality (`rt_hdr_alb_nrm` weights: HDR color
plus albedo and normal), on one 1920x1280 path-traced interior render (not
included; `npubench.testscene` renders a public Cornell box in the same
format). Accuracy is PSNR and SSIM of the display image against the OIDN
library's CPU output. For scale, the noisy input itself scores 20.9 dB.

| Run | ms per frame | Socket W (idle) | NPU W | J per frame (above idle) | PSNR / SSIM |
|-----|-------------:|----------------:|------:|-------------------------:|------------:|
| OIDN library, CPU device, 24 CPUs available | 920 | 15.4 (3.1) | | 14.2 (11.3) | reference |
| Same UNet as ONNX, ONNX Runtime CPU, FP32, 12 threads | 1202 | 16.5 (5.4) | | 19.7 (13.3) | 127.8 dB / 1.0 |
| OIDN library, HIP device on the iGPU | 82.4 | 28.5 (4.3) | | 2.3 (2.0) | 66.5 dB / 0.9999 |
| NPU INT8 A8W8, per-channel weights | 474 | 9.4 (5.7) | 2.2 | 4.4 (1.7) | 31.5 dB / 0.83 (1) |
| NPU INT8 A8W8, per-tensor weights | 473 | 9.4 (5.4) | 2.2 | 4.4 (1.9) | 14.8 dB / 0.77 |
| NPU INT8 A16W8, ConvTranspose upsampling | 6001 | 7.3 (5.0) | 1.0 | 43.7 (13.6) | 14.8 dB / 0.85 (1) |
| NPU INT8 A8W8 per-channel, first and last conv in float | 4292 | 7.1 (5.1) | 0.9 | 30.0 (8.3) | 23.4 dB / 0.83 (1) |
| NPU BF16 | did not compile (2) | | | | |

(1) Scored on ONNX Runtime's CPU run of the same quantized model. For the
per-tensor model, the CPU run and the NPU run scored the same (14.8 dB).
(2) A 256x256 version of the UNet did not finish compiling in 9 minutes.

The NPU INT8 rows run the frame as overlapping 1056x736 tiles with a 96 px
overlap, which on its own changes nothing (the FP32 tiled run also scores
127.8 dB). The ONNX export is the OIDN UNet rebuilt from OIDN's training code,
with OIDN's own input and output processing; it matches the library at
127.8 dB. Two of the 167 nodes of the A8W8 model run on the CPU.

### NPU power mode

The NPU has five power modes (`default`, `powersaver`, `balanced`,
`performance`, `turbo`); `scripts/run.sh xrt-smi examine --report platform`
shows the current one in its "Power Mode" line. All tables above use
`default`. The same three NPU runs, on the idle machine, in the default and the
performance mode (same image, model, cache and settings, 15 minutes apart):

| Model | Mode | ms (median) | Socket W (idle) | NPU W | NPU busy | J per run (above idle) |
|-------|------|------------:|----------------:|------:|---------:|-----------------------:|
| SigLIP 2 base | default | 57.5 | 4.0 (3.1) | 0.52 | 95% | 0.23 (0.05) |
| SigLIP 2 base | performance | 19.1 | 12.8 (4.9) | 4.25 | 92% | 0.24 (0.15) |
| DA3 Small | default | 246 | 3.7 (3.3) | 0.51 | 96% | 0.90 (0.08) |
| DA3 Small | performance | 92.0 | 11.8 (4.7) | 4.18 | 95% | 1.08 (0.65) |
| OIDN A8W8 per-channel, per 1920x1280 frame | default | 474 | 9.4 (4.8) | 2.22 | 50% | 4.44 (2.16) |
| OIDN A8W8 per-channel, per 1920x1280 frame | performance | 475 | 9.5 (5.1) | 2.29 | 50% | 4.53 (2.11) |

The performance mode makes the two transformer models about 3 times faster on
the NPU, for about the same energy per image: the NPU draws 8 times the power
for a third of the time. It does nothing for the OIDN run, where the NPU is
busy only half the time and the CPU's tiling work sets the pace. Even in the
performance mode, the iGPU is faster and the NPU uses less energy per image.

Setting the mode needs root on the host. The driver refuses it from the
rootless container:

```text
$ scripts/run.sh xrt-smi configure --pmode performance
[xrt-smi] ERROR: DRM_IOCTL_AMDXDNA_SET_STATE IOCTL failed (err=-13): Permission denied
```

On the test machine it was set with the distro's own XRT package (Arch Linux
`xrt` 2.21.75), which prints "Power mode is set to performance":

```bash
sudo xrt-smi configure --pmode performance
scripts/run.sh xrt-smi examine --report platform   # Power Mode : performance
sudo xrt-smi configure --pmode default             # back to the default
```

## Known limits

- **BF16 compile time.** The VAIML compiler took 6.5 minutes for SigLIP 2 base
  and 22 minutes for DA3 Small at 504 px, mostly on one CPU core. It did not
  finish a 256x256 version of the OIDN UNet in 9 minutes, and its compile time
  grows with the pixel count, so a 1920x1280 frame is out of reach. Results are
  cached per model and input size, so this is a one-time cost when it
  finishes.
- **INT8 accuracy.** INT8 models compile in seconds, but the OIDN HDR denoiser
  loses too much: 14.8 dB with per-tensor A8W8 weights, worse than the noisy
  input, and 31.5 dB with per-channel weights, against 66.5 dB for OIDN's HIP
  backend. 16-bit activations (A16W8) did not help (14.8 dB). The calibration
  used three renders of a Cornell box scene, not the test image. For
  denoising, use the iGPU.
- **SigLIP 2 BF16 loss.** The NPU's BF16 embeddings are further from FP32
  (mean cosine 0.9962, worst 0.967 on a dark image) than PyTorch's own BF16
  run of the same model on the CPU (mean cosine 0.99974). So most of the loss
  comes from the NPU path, not from bfloat16 itself. The nearest neighbour
  changed for 3 of the 26 images.
- **NPU idle clocking.** In the default power mode, the NPU runs slower when
  the rest of the chip is idle. SigLIP 2 took 39.8 ms per image while the iGPU
  was busy rendering, and 55.7 to 57.5 ms on the quiet machine. DA3 Small took
  181 ms right after a CPU run and 246 ms on an idle chip. So an NPU number
  depends on what else the chip is doing; the tables say which state each
  number comes from. The performance power mode is much faster (19.1 ms for
  SigLIP 2, 92 ms for DA3 Small), but setting it needs root on the host; see
  [NPU power mode](#npu-power-mode).
- **Telemetry.** `npubench.power` reads `gpu_metrics` format 3.0, which Strix
  Point uses; other formats are not decoded. The `system_w` field it records is
  not reliable on the test machine; the tables use socket and NPU power.
- **The EP can load and still run everything on the CPU.** When the NPU
  compile of a subgraph fails, the Vitis AI EP logs the error, skips the
  subgraph ("Catch fatal exception, skip this subgraph"; set
  `XLNX_ENABLE_SKIP_FATAL=0` to stop instead) and ONNX Runtime runs it on the
  CPU. The session still lists `VitisAIExecutionProvider`, so checking the
  providers is not enough. This happened once with DA3 Small: the signature
  files above were deleted during the compile, its last step failed, and all
  857 nodes ran on the CPU at 2.3 s per image. Only the NPU busy counter showed
  it. A result counts only when the NPU's busy counter moved.
- **AMD's quicktest can pass on the CPU.** See [Build and check](#build-and-check).

## Proof that the NPU ran

When the Vitis AI execution provider fails to load, ONNX Runtime silently runs
the whole model on the CPU. `npubench.bench` therefore checks two things on
every NPU run:

1. `VitisAIExecutionProvider` must be in the session's providers. If it is
   not, the command stops before it times anything.
2. The NPU busy counter in `gpu_metrics` must average at least 1% over the
   timed runs. If it does not, the result is still written, with
   `"npu_verified": false`, and the command exits with an error. If the
   counter cannot be read, `npu_verified` is `null`.

It also records how many graph nodes the NPU took (`npu_partition`, from the
EP's report). iGPU runs get the same two checks with the MIGraphX provider and
the GFX busy counter, at 5%. The accuracy tools (`npubench.compare`,
`npubench.oidn_check`) check only the provider, so run them on a model whose
NPU run `npubench.bench` has verified.

## What the image changes, and why

- Base: `ubuntu:24.04`, pinned by digest.
- XRT user space from AMD's two `.deb` files. AMD's amdxdna plugin package also
  carries a DKMS kernel module, firmware and a postinst that runs `rmmod` and
  `modprobe`; only its `/opt/xilinx/xrt` files are unpacked.
- Ryzen AI Software, installed by AMD's own `install_ryzen_ai.sh`, unmodified.
  AMD's archives store files owned by uid 95777, which a rootless build cannot
  map, so every `tar` keeps the extracting user (`TAR_OPTIONS=--no-same-owner`).
- `libncurses6`, `libpython3.12t64` and `libgomp1`, which the Vitis AI EP links
  against and a minimal Ubuntu lacks. Without them the EP does not load.
  `libatomic1` is for Mitsuba (`npubench.testscene`).
- `scripts/entrypoint.sh` puts XRT's libraries in front of the Ryzen AI venv's
  in `LD_LIBRARY_PATH`. The venv bundles its own `libxrt_coreutil`, and XRT
  looks for `libxrt_core` next to whichever copy loads first; with the venv's
  first, the NPU runner fails with "No such library .../voe/lib/libxrt_core.so.2".
- `scripts/run.sh` passes only `/dev/accel/accel0`, caps memory (14 GB, no
  swap) and CPU (6), uses the lowest CPU weight, drops all capabilities and sets
  `no-new-privileges`. It mounts the current folder at `/work`.

## Reproduce the numbers

These are the commands behind the tables, with the input file names
shortened. The extra Python packages for
the exports and the test scene go into `cache/pydeps`, so the image stays as
AMD built it; the Ryzen AI venv already has PyTorch, Transformers and ONNX.
`--no-deps` keeps pip from replacing the venv's own NumPy:

```bash
scripts/run.sh pip install --no-cache-dir --no-deps --target /work/cache/pydeps \
    addict==2.4.0 antlr4-python3-runtime==4.9.3 einops==0.8.2 omegaconf==2.3.0 pyyaml==6.0.3 \
    imageio==2.38.1 opencv-python-headless==5.0.0.93 mitsuba==3.9.1 drjit==1.5.0
```

Run each Python command below inside the container with those packages on the
path, for example:

```bash
scripts/run.sh bash -c 'export PYTHONPATH=$PYTHONPATH:/work/cache/pydeps HF_HOME=/work/cache/hf;
    python -m npubench.export siglip2'
```

```bash
# OIDN: rt_hdr_alb_nrm.tza is in the weights/ folder of the OIDN 2.5.1 source release.
python -m npubench.export oidn --weights rt_hdr_alb_nrm.tza --height 736 --width 1056
# Calibration: three Cornell box renders, none of them the test image.
python -m npubench.testscene --camera-x 0 --camera-y 0 --spp 8 --seed 1 --out data/calib/cbox1
python -m npubench.testscene --camera-x 0.6 --camera-y 0.3 --spp 4 --seed 2 --out data/calib/cbox2
python -m npubench.testscene --camera-x -0.5 --camera-y -0.4 --spp 16 --seed 3 --out data/calib/cbox3
python -m npubench.quantize --fp32 models/oidn_rt_hdr_alb_nrm_1056x736.onnx \
    --out models/oidn_rt_hdr_alb_nrm_1056x736_a8w8pc.onnx --scheme a8w8 --per-channel \
    --crop 736 1056 data/calib/*/input.npz
python -m npubench.bench --model models/oidn_rt_hdr_alb_nrm_1056x736_a8w8pc.onnx --device npu \
    --config default --cache-dir cache/npu --frame 1280 1920 --seconds 12 --idle-s 3
python -m npubench.oidn_check --onnx models/oidn_rt_hdr_alb_nrm_1056x736_a8w8pc.onnx --config default \
    --devices cpu,npu --color color.pfm --albedo albedo.pfm --normal normal.pfm \
    --oidn oidn-cpu-out.pfm --display-exposure 6.22

# SigLIP 2, at the Hugging Face revision pinned in npubench/export.py
python -m npubench.export siglip2
python -m npubench.bench --model models/siglip2_base_patch16_224_vision.onnx --device npu --cache-dir cache/npu
python -m npubench.compare --kind siglip2 --model models/siglip2_base_patch16_224_vision.onnx images/*.png

# Depth Anything 3 Small, through devin-lai/Depth-Anything-3-Onnx at commit 9d75a05
git clone https://github.com/devin-lai/Depth-Anything-3-Onnx cache/third_party/Depth-Anything-3-Onnx
git -C cache/third_party/Depth-Anything-3-Onnx checkout 9d75a0527225a0cad939af3e72dd5c24978d2723
python -m npubench.export da3 --exporter cache/third_party/Depth-Anything-3-Onnx --model depth-anything/DA3-SMALL
python -m npubench.bench --model models/DA3-SMALL_504.onnx --device npu --cache-dir cache/npu
python -m npubench.compare --kind da3 --model models/DA3-SMALL_504.onnx images/*.png
```

The OIDN library rows ran `oidnDenoise` from the ROCm image:

```bash
scripts/run-igpu.sh python -m npubench.oidn_cli_bench --out results/oidn_lib_hip.json -- \
    oidnDenoise -d hip -f RT -q high --hdr color.pfm --alb albedo.pfm --nrm normal.pfm -v 2 -n 102
```

and the same with `-d cpu -n 22`. The CPU and iGPU rows used `--device cpu
--threads 12` (with `NPU_CPUS=24`) and `--device igpu` (with `scripts/run-igpu.sh`).

## Repository layout

| Path | What it is |
|------|------------|
| `Containerfile`, `scripts/build.sh` | The NPU image |
| `Containerfile.rocm`, `scripts/run-igpu.sh` | The iGPU baseline image (ROCm, MIGraphX, OIDN with HIP) |
| `scripts/fetch-amd.sh` | Downloads and verifies AMD's files |
| `scripts/run.sh`, `scripts/entrypoint.sh` | Runs a command in the NPU container |
| `scripts/uninstall.sh` | Removes the NPU and iGPU images and their build layers |
| `npubench/bench.py` | Latency and power on the NPU, CPU or iGPU, with the NPU proof |
| `npubench/power.py` | SoC power and busy counters from `gpu_metrics` |
| `npubench/export.py` | ONNX exports of the three models, at pinned revisions |
| `npubench/quantize.py` | INT8 QDQ quantization with ONNX Runtime's tool |
| `npubench/compare.py` | NPU against CPU FP32 accuracy for SigLIP 2 and DA3 |
| `npubench/oidn_unet.py`, `oidn_check.py`, `oidn_cli_bench.py`, `pfm.py`, `testscene.py` | The OIDN UNet in ONNX, its accuracy check, the library timing wrapper, PFM files and a test scene |
| `tests/` | Unit tests that need neither the NPU nor AMD's software: `python -m unittest discover -s tests` |

## Credits and licenses

This repository is Apache-2.0 (see `LICENSE` and `NOTICE`). It builds on:

| Project | Used for | License |
|---------|----------|---------|
| [AMD Ryzen AI Software 1.9.0](https://ryzenai.docs.amd.com/en/latest/linux.html), XRT 2.26.89, amdxdna plugin | The NPU runtime and compilers; downloaded by you, not included | AMD EULA |
| [Intel Open Image Denoise 2.5.1](https://github.com/RenderKit/oidn) | The UNet, its input and output processing (adapted in `npubench/oidn_unet.py`), the reference output and the HIP baseline | Apache-2.0 |
| [OIDN weights](https://github.com/RenderKit/oidn-weights) | `rt_hdr_alb_nrm.tza` | Apache-2.0 |
| [SigLIP 2 base patch16 224](https://huggingface.co/google/siglip2-base-patch16-224), revision `75de2d5` | Image embedding model | Apache-2.0 |
| [Depth Anything 3 Small](https://huggingface.co/depth-anything/DA3-SMALL), revision `e08cab6`, and its [code](https://github.com/ByteDance-Seed/Depth-Anything-3) | Depth model | Apache-2.0 |
| [Depth-Anything-3-Onnx](https://github.com/devin-lai/Depth-Anything-3-Onnx), commit `9d75a05` | DA3 to ONNX export | Apache-2.0 |
| [AMD RyzenAI-SW](https://github.com/amd/RyzenAI-SW) | The BF16 EP configuration (`npubench/vitisai_config_bf16.json`) | MIT |
| [ONNX Runtime](https://github.com/microsoft/onnxruntime) | Inference and INT8 quantization | MIT |
| [AMD ROCm 7.2.4](https://github.com/ROCm/ROCm) and [MIGraphX](https://github.com/ROCm/AMDMIGraphX) | iGPU baseline | MIT and others |
| [Mitsuba 3](https://github.com/mitsuba-renderer/mitsuba3) | The Cornell box test scene and calibration renders | BSD-style |

Check each model's license before you use it: some Depth Anything 3 models,
such as `depth-anything/DA3-LARGE` and `depth-anything/DA3-GIANT`, are under
CC BY-NC 4.0. They were not used here.

Versions and licenses checked on 2026-11-01.

This is an independent project, not affiliated with or endorsed by AMD. AMD,
Ryzen, Radeon and XDNA are trademarks of Advanced Micro Devices, Inc. Other
names are trademarks of their owners.
