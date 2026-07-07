# UniAD base_track_lidar TensorRT 部署流程

> **FP16 数值修复（2026-07-07）**：`velo_update_trt`（track ref_pts 跨帧传播）
> 存在全局坐标 FP16 灾难性抵消，已修复（先在 fp32 算 `l2g_t1 - l2g_t2`）。
> **engine 必须用含此修复的代码重新导出才生效**；旧 engine 不含修复。
> 本机已验证重导出后裸 FP16 10 帧全 finite（脚本 `tools/verify_fp16_all_paths.sh`）。
> 原理与全路径验证矩阵见 `MAPFUSE_FP16_NAN_ANALYSIS.md` §11–§12。

这份文档记录 `stage1_track_map_lidar/base_track_lidar.py` 的部署规划和当前实现状态。
它复用已经跑通的 `base_bevformer_lidar.py` sparse LiDAR 链路：

```text
raw points
  -> C++ hard voxelization + HardSimpleVFE
  -> libspconv sparse ONNX
  -> TensorRT LiDAR backbone+neck engine
  -> TensorRT LiDAR dense track engine
  -> track states / boxes / scores / labels
```

当前阶段已经新增并验证 dense track ONNX/TensorRT 边界和 C++ runtime。为了不影
响已经跑通的 detection 链路，track 使用新的 `uniad_lidar_track` 二进制，原
`uniad_lidar` 保持 detection-only。

## 当前状态

- 已新增部署侧 detector：`UniADTrackLidarTRT`。
- 已新增部署侧 config：
  `UniAD/projects/configs/stage1_track_map_lidar/base_track_lidar_trt_p.py`。
- 已新增 ONNX 导出脚本：`UniAD/tools/export_track_lidar_onnx.py`。
- 已用二轮训练 checkpoint 完成导出、engine 编译和 runtime 验证：
  `UniAD_train/UniAD/projects/work_dirs/stage1_track_map_lidar/base_track_lidar/epoch_2.pth`。
- 已生成并验证：
  `UniAD_train/UniAD/onnx/base_track_lidar_sparse_encoder_epoch2.onnx`、
  `UniAD_train/UniAD/onnx/base_track_lidar_backbone_neck_epoch2.onnx`、
  `UniAD/onnx/base_track_lidar_trt_epoch2.repaired.onnx`、
  `UniAD/engine/base_track_lidar_backbone_neck_epoch2.engine`、
  `UniAD/engine/base_track_lidar_track_head_epoch2.engine`。
- 已新增 C++ runtime：`inference_app/sparse_lidar/build/uniad_lidar_track`。
- 已用 `epoch_2.pth` 跑通 10 帧 raw-points runtime，输出 track state、
  detection txt、带 track id 的 BEV SVG 和 WebM 视频。

## 命名约定

- `UniAD_train/UniAD`：训练侧代码，导出 sparse encoder ONNX、backbone+neck
  ONNX，准备部署输入数据。
- `UniAD`：部署侧代码，导出 dense track TensorRT 边界 ONNX。
- `inference_app/enqueueV3`：TensorRT 10.x plugin 工程，生成
  `libuniad_plugin.so`。
- `inference_app/sparse_lidar`：LiDAR C++ runtime。`uniad_lidar` 是 detection
  runtime，`uniad_lidar_track` 是 stage1 track runtime。

## 0. 环境

以下命令默认从仓库根目录执行：

```bash
cd /home/baojiali/Downloads/public_code/DL4AGX/AV-Solutions/uniad-trt
conda activate uniad_train

export TRT_PATH=/home/baojiali/Downloads/TensorRT-10.7.0.23
export PATH=$TRT_PATH/bin:$PATH
export LD_LIBRARY_PATH=$TRT_PATH/lib:$LD_LIBRARY_PATH

export TARGET_GPU_SM=89
export SPCONV_CUDA_VERSION=11.4
export UNIAD_TRAIN_DIR=$PWD/UniAD_train/UniAD
export TRACK_CFG=$UNIAD_TRAIN_DIR/projects/configs/stage1_track_map_lidar/base_track_lidar.py
export TRACK_TRT_CFG=$PWD/UniAD/projects/configs/stage1_track_map_lidar/base_track_lidar_trt_p.py
export CKPT=$UNIAD_TRAIN_DIR/projects/work_dirs/stage1_track_map_lidar/base_track_lidar/latest.pth
export TAG=epoch2
export NUM_FRAMES=10
```

如果只训练了一轮、还没有 `latest.pth`，把 `CKPT` 改成：

```bash
export CKPT=$UNIAD_TRAIN_DIR/projects/work_dirs/stage1_track_map_lidar/base_track_lidar/epoch_1.pth
```

如果 `CKPT` 不存在，先不要做 engine 质量验证；最多只能做导出脚本/配置的语法级
smoke test。

## 1. 准备部署输入数据

这一步只读 dataset，写 raw points、metadata 和可选 GT，不加载 checkpoint。
metadata 里需要保留 `timestamp`、`ego2global`、`prev_bev_exists` 和
`ego_motion_delta`，后续 C++ track runtime 会用它们更新 prev-BEV 和 track pose。

```bash
cd UniAD_train/UniAD

python tools/prepare_bevformer_lidar_deploy_data.py \
  --config projects/configs/stage1_track_map_lidar/base_track_lidar.py \
  --split test \
  --start-index 0 \
  --max-frames $NUM_FRAMES \
  --out-dir dumped_inputs/base_track_lidar_deploy_data_${NUM_FRAMES}f

cd -
```

输出目录结构和 `bevformer_lidar_deploy_data` 一致：

```text
raw_points_000000.bin
raw_points_000001.bin
img_metas_000000.json
img_metas_000001.json
gt_detections_000000.txt
manifest.json
```

## 2. 导出 ONNX

### 2.1 Sparse Encoder ONNX

```bash
cd UniAD_train/UniAD

python tools/export_bevformer_lidar_sparse_onnx.py \
  $TRACK_CFG \
  $CKPT \
  --split test \
  --index 0 \
  --onnx-file onnx/base_track_lidar_sparse_encoder_${TAG}.onnx \
  --tensor-prefix dumped_inputs/base_track_lidar_sparse_encoder_${TAG}/infer

cd -
```

注意：track dataset 的 queue length 是 5，当前 sparse ONNX 可能导出固定 batch=5
的 sparse dense 输出。`uniad_lidar_track` 在 raw-points 单帧运行时会检测
`B * 1x256x120x160` 这种输出，并取 batch 0 喂给 backbone+neck engine。

### 2.2 LiDAR Backbone+Neck ONNX

```bash
cd UniAD_train/UniAD

python tools/export_bevformer_lidar_backbone_neck_onnx.py \
  $TRACK_CFG \
  $CKPT \
  --onnx-file onnx/base_track_lidar_backbone_neck_${TAG}.onnx

cd -
```

### 2.3 Dense Track ONNX

Dense track ONNX 从 `lidar_bev` 开始，不包含 sparse encoder 和 backbone+neck。
输入包括上一帧 `prev_bev`、BEV shift、track state、timestamp、lidar-to-global
pose 和 `max_obj_id`。

```bash
cd UniAD

python tools/export_track_lidar_onnx.py \
  projects/configs/stage1_track_map_lidar/base_track_lidar_trt_p.py \
  $CKPT \
  --onnx-file onnx/base_track_lidar_trt_${TAG}.onnx \
  --track-state-len 601 \
  --dump-input-dir dumped_inputs/base_track_lidar_trt_trace_${TAG}

cd -
```

输出：

```text
UniAD/onnx/base_track_lidar_trt_${TAG}.onnx
UniAD/onnx/base_track_lidar_trt_${TAG}.repaired.onnx
UniAD/dumped_inputs/base_track_lidar_trt_trace_${TAG}/*.dat
```

## 3. 编译 C++ 和 TensorRT Plugin

```bash
cmake -S inference_app/enqueueV3 -B inference_app/enqueueV3/build \
  -DTENSORRT_PATH=$TRT_PATH \
  -DTARGET_GPU_SM=$TARGET_GPU_SM

cmake --build inference_app/enqueueV3/build -j$(nproc)
```

如果需要重新编译 sparse LiDAR runtime：

```bash
cmake -S inference_app/sparse_lidar -B inference_app/sparse_lidar/build \
  -DSPCONV_CUDA_VERSION=$SPCONV_CUDA_VERSION \
  -DTENSORRT_PATH=$TRT_PATH

cmake --build inference_app/sparse_lidar/build -j$(nproc)
```

成功后会生成：

```text
inference_app/sparse_lidar/build/uniad_lidar
inference_app/sparse_lidar/build/uniad_lidar_track
```

## 4. 编译 TensorRT Engine

先确认已经执行过第 0 节里的 TensorRT 环境变量，尤其是
`LD_LIBRARY_PATH=$TRT_PATH/lib:$LD_LIBRARY_PATH`。否则 `trtexec` 可能找不到
`libnvinfer_plugin.so.10`。

```bash
mkdir -p UniAD/engine
```

### 4.1 Backbone+Neck Engine

```bash
$TRT_PATH/bin/trtexec \
  --onnx=UniAD_train/UniAD/onnx/base_track_lidar_backbone_neck_${TAG}.onnx \
  --saveEngine=UniAD/engine/base_track_lidar_backbone_neck_${TAG}.engine \
  --fp16 \
  --skipInference
```

### 4.2 Dense Track Engine

`base_track_lidar.py` 有 600 个 object query 和 1 个 SDC query，所以首帧最小
track state 长度是 `601`。后续帧会把初始 query 和 active tracks 拼起来，默认
先给到 `MAX=901`；如果 runtime 报 track state 超过 profile，就增大 `MAX`
重新编译 engine。

```bash
MIN=601
OPT=601
MAX=901

# ONNX 会裁掉未使用的 prev_track_intances2/7/10；profile 只写实际输入。
TRACK_SHAPES=prev_track_intances0:Lx512,prev_track_intances1:Lx3,prev_track_intances3:L,prev_track_intances4:L,prev_track_intances5:L,prev_track_intances6:L,prev_track_intances8:L,prev_track_intances9:Lx10,prev_track_intances11:Lx4x256,prev_track_intances12:Lx4,prev_track_intances13:L

$TRT_PATH/bin/trtexec \
  --onnx=UniAD/onnx/base_track_lidar_trt_${TAG}.repaired.onnx \
  --saveEngine=UniAD/engine/base_track_lidar_track_head_${TAG}.engine \
  --staticPlugins=inference_app/enqueueV3/build/libuniad_plugin.so \
  --fp16 \
  --minShapes=${TRACK_SHAPES//L/${MIN}} \
  --optShapes=${TRACK_SHAPES//L/${OPT}} \
  --maxShapes=${TRACK_SHAPES//L/${MAX}} \
  --skipInference
```

固定形状输入由 ONNX 决定：

```text
lidar_bev:      1x256x120x160
prev_bev:       1x256x120x160
shift:          1x2
use_prev_bev:   1
prev_timestamp: 1
timestamp:      1
prev_l2g_r_mat: 1x3x3
l2g_r_mat:      1x3x3
prev_l2g_t:     1x3
l2g_t:          1x3
max_obj_id:     1
```

## 5. C++ Runtime 运行

`uniad_lidar_track` 从 raw points 开始跑完整链路，并维护：

- `prev_bev`：来自 dense track engine 的 `bev_embed` 输出。
- 11 个实际 ONNX 输入 state：`prev_track_intances0/1/3/4/5/6/8/9/11/12/13`。
- `prev_timestamp/prev_l2g_r_mat/prev_l2g_t/max_obj_id`。

首帧和 scene reset 会从 `UniAD/dumped_inputs/base_track_lidar_trt_trace_${TAG}`
恢复初始 state。这个目录由 dense track ONNX export 生成，不能随便用全零替代，
因为 `prev_track_intances0/1` 里包含 learned query embedding 和 reference points。

```bash
export TRT_PATH=/home/baojiali/Downloads/TensorRT-10.7.0.23
export LD_LIBRARY_PATH=$TRT_PATH/lib:$LD_LIBRARY_PATH
export CUDA_VISIBLE_DEVICES=0

rm -rf UniAD/output/base_track_lidar_track_${TAG}_${NUM_FRAMES}f

inference_app/sparse_lidar/build/uniad_lidar_track \
  UniAD_train/UniAD/onnx/base_track_lidar_sparse_encoder_${TAG}.onnx \
  UniAD/engine/base_track_lidar_backbone_neck_${TAG}.engine \
  UniAD/engine/base_track_lidar_track_head_${TAG}.engine \
  inference_app/enqueueV3/build/libuniad_plugin.so \
  UniAD_train/UniAD/dumped_inputs/base_track_lidar_deploy_data_${NUM_FRAMES}f \
  UniAD/output/base_track_lidar_track_${TAG}_${NUM_FRAMES}f \
  $NUM_FRAMES \
  --metadata-json UniAD_train/UniAD/dumped_inputs/base_track_lidar_deploy_data_${NUM_FRAMES}f \
  --gt-detections UniAD_train/UniAD/dumped_inputs/base_track_lidar_deploy_data_${NUM_FRAMES}f \
  --track-init-dir UniAD/dumped_inputs/base_track_lidar_trt_trace_${TAG} \
  --score-threshold 0.2 \
  --bev-max-draw 200
```

runtime 会从 metadata 读取：

```text
timestamp
ego2global
prev_bev_exists
ego_motion_delta
```

`timestamp` 在 JSON 里是绝对 epoch 秒，数值约 1.7e9。直接转 float32 会丢掉亚秒
级时间差，所以 C++ 侧喂给 engine 的是以当前 scene 首帧为 0 的相对 timestamp。
模型只使用 `timestamp - prev_timestamp`，因此这个处理和原图等价。

验证过的输出示例：

```text
frame_000000_detections.txt
frame_000000_bev.svg
frame_000000_bev_compare.svg
frame_000000_prev_track_intances0_out.bin
frame_000000_bev_embed.bin
frame_000001_detections.txt
frame_000001_bev.svg
frame_000001_bev_compare.svg
```

`frame_*_bev.svg` 会按 track id 固定上色，并用醒目的 `ID <track_id>` 标签标出
每个 track；`frame_*_bev_compare.svg` 左侧是 GT boxes，右侧是带 track id 的
TRT prediction。

### 5.1 生成 Tracking WebM

runtime 每帧都会生成：

```text
frame_000000_bev.svg
frame_000000_bev_compare.svg
```

`frame_*_bev.svg` 是 TRT tracking prediction；`frame_*_bev_compare.svg` 左侧是
GT boxes，右侧是 TRT tracking prediction。两种 SVG 里的 prediction 都按 track
id 固定上色，并用醒目的 `ID <track_id>` 标签标出每个 track。

如果机器上有 `google-chrome` 和 `ffmpeg`，可以直接把连续帧合成 WebM：

```bash
OUT=UniAD/output/base_track_lidar_track_${TAG}_${NUM_FRAMES}f
mkdir -p $OUT/video_frames_pred $OUT/video_frames_compare
export OUT

python3 - <<'PY'
import os
import pathlib
import subprocess

out_dir = pathlib.Path(os.environ["OUT"]).resolve()
chrome = "/usr/bin/google-chrome"
jobs = [
    ("frame_*_bev.svg", out_dir / "video_frames_pred", "900,1200"),
    ("frame_*_bev_compare.svg", out_dir / "video_frames_compare", "1836,1252"),
]
for pattern, frame_dir, window_size in jobs:
    frame_dir.mkdir(parents=True, exist_ok=True)
    svgs = sorted(out_dir.glob(pattern))
    if not svgs:
        raise SystemExit(f"No SVG frames found for {pattern}")
    for i, svg in enumerate(svgs):
        png = frame_dir / f"frame_{i:06d}.png"
        subprocess.run([
            chrome,
            "--headless=new",
            "--disable-gpu",
            "--no-sandbox",
            "--hide-scrollbars",
            f"--window-size={window_size}",
            f"--screenshot={png}",
            svg.as_uri(),
        ], check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
PY

ffmpeg -y -framerate 2 -i $OUT/video_frames_pred/frame_%06d.png \
  -c:v libvpx-vp9 -pix_fmt yuv420p -b:v 0 -crf 30 \
  $OUT/base_track_lidar_tracking_pred_${NUM_FRAMES}f_${TAG}.webm

ffmpeg -y -framerate 2 -i $OUT/video_frames_compare/frame_%06d.png \
  -c:v libvpx-vp9 -pix_fmt yuv420p -b:v 0 -crf 30 \
  $OUT/base_track_lidar_tracking_compare_${NUM_FRAMES}f_${TAG}.webm
```

prediction-only 视频更适合看 track id 连续性；compare 视频更适合看 GT 和
prediction 的位置差异。

`epoch_2.pth` 的 10 帧验证输出：

```text
frame 0: 4 tracks, max_obj_id=4
frame 1: 4 tracks, max_obj_id=4
frame 2: 4 tracks, max_obj_id=4
frame 3: 4 tracks, max_obj_id=4
frame 4: 4 tracks, max_obj_id=4
frame 5: 5 tracks, max_obj_id=5
frame 6: 5 tracks, max_obj_id=5
frame 7: 5 tracks, max_obj_id=5
frame 8: 5 tracks, max_obj_id=5
frame 9: 5 tracks, max_obj_id=5
```

输出目录：

```text
UniAD/output/base_track_lidar_track_epoch2_10f
```

生成的视频：

```text
UniAD/output/base_track_lidar_track_epoch2_10f/base_track_lidar_tracking_pred_10f_epoch2.webm
UniAD/output/base_track_lidar_track_epoch2_10f/base_track_lidar_tracking_compare_10f_epoch2.webm
```

## 6. 已验证命令

本文档中的 epoch2 命令已经按顺序验证过：

- `prepare_bevformer_lidar_deploy_data.py` 生成 10 帧 raw points / metadata / GT。
- `export_bevformer_lidar_sparse_onnx.py` 导出
  `base_track_lidar_sparse_encoder_epoch2.onnx`，summary 中
  `dense_bev_shape=[5, 256, 120, 160]`。
- `export_bevformer_lidar_backbone_neck_onnx.py` 导出
  `base_track_lidar_backbone_neck_epoch2.onnx`，输入输出都是
  `1x256x120x160`。
- `export_track_lidar_onnx.py` 导出并 repair
  `base_track_lidar_trt_epoch2.repaired.onnx`。
- `trtexec --skipInference` 成功编译
  `base_track_lidar_backbone_neck_epoch2.engine` 和
  `base_track_lidar_track_head_epoch2.engine`。
- `uniad_lidar_track` 跑通 10 帧 raw-points runtime。
- `google-chrome` + `ffmpeg` 生成 10 帧 prediction-only 和 GT-compare WebM。

```bash
python3 -m py_compile \
  UniAD/tools/export_track_lidar_onnx.py \
  UniAD/projects/configs/stage1_track_map_lidar/base_track_lidar_trt_p.py \
  UniAD/projects/mmdet3d_plugin/uniad/detectors/uniad_track_lidar.py \
  UniAD/projects/mmdet3d_plugin/uniad/dense_heads/bevformer_lidar_head.py

cmake --build inference_app/sparse_lidar/build --target uniad_lidar_track -j$(nproc)
```

TensorRT 10.7 编译时会输出一些 deprecated plugin API warning，目前不影响
`uniad_lidar_track` 编译和运行。
