# UniAD base_e2e_lidar TensorRT 部署流程

这份文档记录 `stage2_e2e_lidar/base_e2e_lidar.py` 的 LiDAR E2E 部署链路。
它在 `base_track_lidar.py` 的 tracking 链路后面接上 MotionHead：

```text
raw points
  -> C++ hard voxelization + HardSimpleVFE
  -> libspconv sparse ONNX
  -> TensorRT LiDAR backbone+neck engine
  -> TensorRT LiDAR dense E2E engine
  -> track states / boxes / scores / labels / trajectories
```

sparse encoder 的 ONNX 由 `libspconv` 直接加载，不编译成 TensorRT engine。
TensorRT engine 只有两段：`backbone+neck` 和 dense E2E `track+motion`。

## 当前状态

已用 `stage2_e2e_lidar/base_e2e_lidar/epoch_2.pth` 完成同 checkpoint 部署验证：

- `latest.pth` 在验证时指向 `epoch_2.pth`，但训练日志已经继续进入 epoch 3。
  为了复现，本文件命令默认显式使用 `epoch_2.pth`。
- 已导出 stage2 sparse encoder ONNX：
  `UniAD_train/UniAD/onnx/base_e2e_lidar_sparse_encoder_epoch2.onnx`。
- 已导出 stage2 backbone+neck ONNX：
  `UniAD_train/UniAD/onnx/base_e2e_lidar_backbone_neck_epoch2.onnx`。
- 已导出 dense E2E ONNX：
  `UniAD/onnx/base_e2e_lidar_trt_epoch2.repaired.onnx`。
- 已用 TensorRT 10.7 编译：
  `UniAD/engine/base_e2e_lidar_backbone_neck_epoch2.engine` 和
  `UniAD/engine/base_e2e_lidar_trt_epoch2.engine`。
- 已用 `uniad_lidar_e2e` 跑通 10 帧 raw-points runtime：
  `UniAD/output/base_e2e_lidar_epoch2_10f`。

相关实测日志：

```text
UniAD/logs/trtexec_base_e2e_lidar_backbone_neck_epoch2.log
UniAD/logs/trtexec_base_e2e_lidar_trt_epoch2.log
UniAD/logs/trtexec_base_e2e_lidar_trt_epoch2_infer.log
UniAD/logs/uniad_lidar_e2e_epoch2_10f.log
```

## 命名约定

- `UniAD_train/UniAD`：训练侧代码，导出 sparse encoder ONNX、backbone+neck
  ONNX，并准备 raw-points 部署输入数据。
- `UniAD`：部署侧代码，导出 dense E2E TensorRT 边界 ONNX。
- `inference_app/enqueueV3`：TensorRT 10.x plugin 工程，生成
  `libuniad_plugin.so`。
- `inference_app/sparse_lidar`：LiDAR C++ runtime。`uniad_lidar_e2e` 是
  stage2 E2E runtime。
- `TAG=epoch2`：和 checkpoint 绑定的产物后缀。换 checkpoint 时同步改
  `CKPT` 和 `TAG`，不要只改其中一个。

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
export E2E_CFG=$UNIAD_TRAIN_DIR/projects/configs/stage2_e2e_lidar/base_e2e_lidar.py
export E2E_TRT_CFG=$PWD/UniAD/projects/configs/stage2_e2e_lidar/base_e2e_lidar_trt_p.py
export CKPT=$UNIAD_TRAIN_DIR/projects/work_dirs/stage2_e2e_lidar/base_e2e_lidar/epoch_2.pth
export TAG=epoch2
export NUM_FRAMES=10

# 如果训练还在占 GPU，选一张有余量的卡做导出。
export EXPORT_GPU=2
```

`base_e2e_lidar.py` 的 motion head 需要
`data/others/motion_anchor_infos_kl.pkl`。部署侧导出时也要能从 `UniAD`
目录下读到它。本机已经建好：

```text
UniAD/data/others/motion_anchor_infos_kl.pkl
```

如果缺失，按实际文件位置创建软链，例如：

```bash
mkdir -p UniAD/data/others
ln -sf ../../../UniAD_kl_train/UniAD/data/others/motion_anchor_infos_kl.pkl \
  UniAD/data/others/motion_anchor_infos_kl.pkl
```

## 1. 准备部署输入数据

这一步只读 dataset，写 raw points、metadata 和可选 GT，不加载 checkpoint。
metadata 里的 `timestamp`、`ego2global`、`prev_bev_exists` 和
`ego_motion_delta` 会被 C++ runtime 用来维护 prev-BEV、track state 和 ego-motion
对齐。

```bash
cd UniAD_train/UniAD

python tools/prepare_bevformer_lidar_deploy_data.py \
  --config $E2E_CFG \
  --split test \
  --start-index 0 \
  --max-frames $NUM_FRAMES \
  --out-dir dumped_inputs/base_e2e_lidar_deploy_data_${NUM_FRAMES}f

cd -
```

输出示例：

```text
raw_points_000000.bin
raw_points_000001.bin
img_metas_000000.json
img_metas_000001.json
gt_detections_000000.txt
manifest.json
```

本机已验证目录：

```text
UniAD_train/UniAD/dumped_inputs/base_e2e_lidar_deploy_data_10f
```

## 2. 导出 ONNX

三段 ONNX 都必须来自同一个 checkpoint。`base_e2e_lidar.py` 虽然设置了
`freeze_lidar_backbone=True`，checkpoint 里仍然包含 sparse encoder 和
backbone+neck 权重，所以前端也应从 stage2 checkpoint 导出。

### 2.1 Sparse Encoder ONNX

```bash
cd UniAD_train/UniAD

CUDA_VISIBLE_DEVICES=$EXPORT_GPU python tools/export_bevformer_lidar_sparse_onnx.py \
  $E2E_CFG \
  $CKPT \
  --split test \
  --index 0 \
  --onnx-file onnx/base_e2e_lidar_sparse_encoder_${TAG}.onnx \
  --tensor-prefix dumped_inputs/base_e2e_lidar_sparse_encoder_${TAG}/infer

cd -
```

本机 `epoch2` 导出结果：

```text
UniAD_train/UniAD/onnx/base_e2e_lidar_sparse_encoder_epoch2.onnx
UniAD_train/UniAD/dumped_inputs/base_e2e_lidar_sparse_encoder_epoch2/infer.[voxels|coors|dense|info]
```

导出 summary 中 sparse dense 输出是 `5x256x120x160`。这是因为 stage2 继承了
tracking 数据队列；`uniad_lidar_e2e` 在 raw-points 单帧运行时会检测这种
`5 * 1x256x120x160` 输出，并取 batch 0 继续跑。

### 2.2 LiDAR Backbone+Neck ONNX

```bash
cd UniAD_train/UniAD

CUDA_VISIBLE_DEVICES=$EXPORT_GPU python tools/export_bevformer_lidar_backbone_neck_onnx.py \
  $E2E_CFG \
  $CKPT \
  --onnx-file onnx/base_e2e_lidar_backbone_neck_${TAG}.onnx

cd -
```

本机 `epoch2` 导出结果：

```text
UniAD_train/UniAD/onnx/base_e2e_lidar_backbone_neck_epoch2.onnx
UniAD_train/UniAD/onnx/base_e2e_lidar_backbone_neck_epoch2.summary.json
```

输入输出都是 `1x256x120x160`。

### 2.3 Dense E2E ONNX

Dense E2E ONNX 从 `lidar_bev` 开始，不包含 sparse encoder 和 backbone+neck。
它包括 tracking state 更新、detection 输出和 MotionHead trajectory 输出。

```bash
cd UniAD

CUDA_VISIBLE_DEVICES=$EXPORT_GPU python tools/export_e2e_lidar_onnx.py \
  projects/configs/stage2_e2e_lidar/base_e2e_lidar_trt_p.py \
  $CKPT \
  --onnx-file onnx/base_e2e_lidar_trt_${TAG}.onnx \
  --track-state-len 601 \
  --dump-input-dir dumped_inputs/base_e2e_lidar_trt_trace_${TAG}

cd -
```

输出：

```text
UniAD/onnx/base_e2e_lidar_trt_${TAG}.onnx
UniAD/onnx/base_e2e_lidar_trt_${TAG}.repaired.onnx
UniAD/dumped_inputs/base_e2e_lidar_trt_trace_${TAG}/*.dat
```

`dumped_inputs/base_e2e_lidar_trt_trace_${TAG}` 是 C++ runtime 的初始 track state。
不要用全零替代，因为 `prev_track_intances0/1` 里有 learned query embedding 和
reference points。

dense E2E 导出时，checkpoint 里 sparse/backbone/neck/criterion 权重会显示为
`unexpected key`，这是预期现象，因为部署侧 dense 边界不接收这些模块。实测
`epoch2` 导出没有看到 dense track/motion 权重的 missing key。

ONNX 会裁掉未使用的 `prev_track_intances2/7/10`，实际 TensorRT profile 只需要
11 个 track-state 输入。

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
inference_app/enqueueV3/build/libuniad_plugin.so
inference_app/sparse_lidar/build/uniad_lidar_e2e
```

## 4. 编译 TensorRT Engine

### 4.1 Backbone+Neck Engine

```bash
mkdir -p UniAD/engine UniAD/logs

$TRT_PATH/bin/trtexec \
  --onnx=UniAD_train/UniAD/onnx/base_e2e_lidar_backbone_neck_${TAG}.onnx \
  --saveEngine=UniAD/engine/base_e2e_lidar_backbone_neck_${TAG}.engine \
  --fp16 \
  --skipInference \
  2>&1 | tee UniAD/logs/trtexec_base_e2e_lidar_backbone_neck_${TAG}.log
```

本机 `epoch2` 结果：

```text
PASSED TensorRT.trtexec [TensorRT v100700]
Created engine with size: 8.879 MiB
```

### 4.2 Dense E2E Engine

`base_e2e_lidar.py` 初始 track state 长度是 `601`。后续帧会把 active tracks
拼回 state，本文档实测先给 `MAX=1201`。如果 runtime 报 track state 超过 profile，
增大 `MAX` 后重新编译 dense E2E engine。

```bash
MIN=601
OPT=601
MAX=1201

TRACK_SHAPES=prev_track_intances0:Lx512,prev_track_intances1:Lx3,prev_track_intances3:L,prev_track_intances4:L,prev_track_intances5:L,prev_track_intances6:L,prev_track_intances8:L,prev_track_intances9:Lx10,prev_track_intances11:Lx4x256,prev_track_intances12:Lx4,prev_track_intances13:L

$TRT_PATH/bin/trtexec \
  --onnx=UniAD/onnx/base_e2e_lidar_trt_${TAG}.repaired.onnx \
  --saveEngine=UniAD/engine/base_e2e_lidar_trt_${TAG}.engine \
  --staticPlugins=inference_app/enqueueV3/build/libuniad_plugin.so \
  --fp16 \
  --minShapes=${TRACK_SHAPES//L/${MIN}} \
  --optShapes=${TRACK_SHAPES//L/${OPT}} \
  --maxShapes=${TRACK_SHAPES//L/${MAX}} \
  --skipInference \
  2>&1 | tee UniAD/logs/trtexec_base_e2e_lidar_trt_${TAG}.log
```

本机 `epoch2` 结果：

```text
PASSED TensorRT.trtexec [TensorRT v100700]
Detected 22 inputs and 28 output network tensors.
Created engine with size: 79.0732 MiB
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

### 4.3 Dense E2E Engine 加载检查

```bash
TRACK_SHAPES=prev_track_intances0:601x512,prev_track_intances1:601x3,prev_track_intances3:601,prev_track_intances4:601,prev_track_intances5:601,prev_track_intances6:601,prev_track_intances8:601,prev_track_intances9:601x10,prev_track_intances11:601x4x256,prev_track_intances12:601x4,prev_track_intances13:601

$TRT_PATH/bin/trtexec \
  --loadEngine=UniAD/engine/base_e2e_lidar_trt_${TAG}.engine \
  --staticPlugins=inference_app/enqueueV3/build/libuniad_plugin.so \
  --shapes=${TRACK_SHAPES} \
  --duration=1 \
  --warmUp=100 \
  --iterations=10 \
  2>&1 | tee UniAD/logs/trtexec_base_e2e_lidar_trt_${TAG}_infer.log
```

本机 `epoch2` 检查通过。日志中可以看到 motion 输出都走动态分配：

```text
traj_scores_0 is dynamic and will be created during execution using OutputAllocator.
traj_0 is dynamic and will be created during execution using OutputAllocator.
traj_scores_1 is dynamic and will be created during execution using OutputAllocator.
traj_1 is dynamic and will be created during execution using OutputAllocator.
traj_scores is dynamic and will be created during execution using OutputAllocator.
traj is dynamic and will be created during execution using OutputAllocator.
valid_traj_masks is dynamic and will be created during execution using OutputAllocator.
```

随机输入下本机 `trtexec` 统计只用于确认 engine 可加载，不代表端到端性能：

```text
Throughput: 23.148 qps
Latency median: 20.2985 ms
GPU Compute Time median: 16.4465 ms
```

## 5. C++ Runtime 运行

`uniad_lidar_e2e` 从 raw points 开始跑完整链路，并维护：

- `prev_bev`：来自 dense E2E engine 的 `bev_embed` 输出。
- 11 个实际 ONNX 输入 state：
  `prev_track_intances0/1/3/4/5/6/8/9/11/12/13`。
- `prev_timestamp/prev_l2g_r_mat/prev_l2g_t/max_obj_id`。
- MotionHead 输出：
  `traj_scores_0/traj_0/traj_scores_1/traj_1/traj_scores/traj/valid_traj_masks`。

运行命令：

```bash
export LD_LIBRARY_PATH=$TRT_PATH/lib:$LD_LIBRARY_PATH

rm -rf UniAD/output/base_e2e_lidar_${TAG}_${NUM_FRAMES}f

inference_app/sparse_lidar/build/uniad_lidar_e2e \
  UniAD_train/UniAD/onnx/base_e2e_lidar_sparse_encoder_${TAG}.onnx \
  UniAD/engine/base_e2e_lidar_backbone_neck_${TAG}.engine \
  UniAD/engine/base_e2e_lidar_trt_${TAG}.engine \
  inference_app/enqueueV3/build/libuniad_plugin.so \
  UniAD_train/UniAD/dumped_inputs/base_e2e_lidar_deploy_data_${NUM_FRAMES}f \
  UniAD/output/base_e2e_lidar_${TAG}_${NUM_FRAMES}f \
  $NUM_FRAMES \
  --metadata-json UniAD_train/UniAD/dumped_inputs/base_e2e_lidar_deploy_data_${NUM_FRAMES}f \
  --gt-detections UniAD_train/UniAD/dumped_inputs/base_e2e_lidar_deploy_data_${NUM_FRAMES}f \
  --track-init-dir UniAD/dumped_inputs/base_e2e_lidar_trt_trace_${TAG} \
  --track-state-len 601 \
  --max-track-state-len 1201 \
  --score-threshold 0.2 \
  --bev-max-draw 200 \
  2>&1 | tee UniAD/logs/uniad_lidar_e2e_${TAG}_${NUM_FRAMES}f.log
```

输出示例：

```text
frame_000000_detections.txt
frame_000000_bev.svg
frame_000000_bev_compare.svg
frame_000000_lidar_bev.bin
frame_000000_bev_embed.bin
frame_000000_prev_track_intances0_out.bin
frame_000000_traj_scores_0.bin
frame_000000_traj_0.bin
frame_000000_traj_scores_1.bin
frame_000000_traj_1.bin
frame_000000_traj_scores.bin
frame_000000_traj.bin
frame_000000_valid_traj_masks.bin
```

`frame_*_bev.svg` 会按 track id 固定上色；`frame_*_bev_compare.svg` 左侧是 GT
boxes，右侧是 TRT prediction。

本机 `epoch2` 的 10 帧验证结果：

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
UniAD/output/base_e2e_lidar_epoch2_10f
```

### 5.1 生成 WebM 可视化

runtime 每帧都会生成：

```text
frame_000000_bev.svg
frame_000000_bev_compare.svg
```

`frame_*_bev.svg` 是 TRT E2E prediction；`frame_*_bev_compare.svg` 左侧是 GT
boxes，右侧是 TRT prediction。可以用 `google-chrome` 把 SVG 渲成 PNG，再用
`ffmpeg` 合成 WebM：

```bash
OUT=UniAD/output/base_e2e_lidar_${TAG}_${NUM_FRAMES}f
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
  $OUT/base_e2e_lidar_pred_${NUM_FRAMES}f_${TAG}.webm

ffmpeg -y -framerate 2 -i $OUT/video_frames_compare/frame_%06d.png \
  -c:v libvpx-vp9 -pix_fmt yuv420p -b:v 0 -crf 30 \
  $OUT/base_e2e_lidar_compare_${NUM_FRAMES}f_${TAG}.webm
```

本机 `epoch2` 已生成：

```text
UniAD/output/base_e2e_lidar_epoch2_10f/base_e2e_lidar_pred_10f_epoch2.webm
UniAD/output/base_e2e_lidar_epoch2_10f/base_e2e_lidar_compare_10f_epoch2.webm
```

### 5.2 生成 MotionHead 轨迹可视化

`base_e2e_lidar.py` 里的 `motion_head.type='MotionHeadLidar'` 输出多模态未来
轨迹。部署侧 dense E2E engine 会写出：

```text
traj_scores_0 / traj_0
traj_scores_1 / traj_1
traj_scores   / traj
valid_traj_masks
```

其中 `traj_scores/traj` 是最后一个 decoder layer 的结果；`traj_0` 和 `traj_1`
是前两个 decoder layer 的中间结果。`traj_scores` 形状是：

```text
num_tracks x 6
```

`traj` 形状是：

```text
num_tracks x 6 x 12 x 5
```

含义是：每个 active track 有 6 个未来轨迹 mode，每个 mode 预测未来 12 步，每步
5 个参数。前两个参数是未来位移 `dx, dy`，后 3 个参数是 bivariate Gaussian 的
`sigma_x, sigma_y, rho`。可视化时把 `dx, dy` 加到当前 detection center 上。

生成带 motion trajectory 的 SVG：

```bash
python3 inference_app/sparse_lidar/visualize_e2e_motion.py \
  UniAD/output/base_e2e_lidar_${TAG}_${NUM_FRAMES}f \
  --gt-dir UniAD_train/UniAD/dumped_inputs/base_e2e_lidar_deploy_data_${NUM_FRAMES}f \
  --num-frames $NUM_FRAMES \
  --top-modes 6 \
  --min-score 0.2
```

输出：

```text
frame_000000_motion.svg
frame_000000_motion_compare.svg
```

图里粗线是 `traj_scores` 最高的 mode，淡色虚线是其它候选 mode。

合成 motion WebM：

```bash
OUT=UniAD/output/base_e2e_lidar_${TAG}_${NUM_FRAMES}f
mkdir -p $OUT/video_frames_motion $OUT/video_frames_motion_compare
export OUT

python3 - <<'PY'
import os
import pathlib
import subprocess

out_dir = pathlib.Path(os.environ["OUT"]).resolve()
chrome = "/usr/bin/google-chrome"
jobs = [
    ("frame_*_motion.svg", out_dir / "video_frames_motion", "900,1200"),
    ("frame_*_motion_compare.svg", out_dir / "video_frames_motion_compare", "1836,1252"),
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

ffmpeg -y -framerate 2 -i $OUT/video_frames_motion/frame_%06d.png \
  -c:v libvpx-vp9 -pix_fmt yuv420p -b:v 0 -crf 30 \
  $OUT/base_e2e_lidar_motion_${NUM_FRAMES}f_${TAG}.webm

ffmpeg -y -framerate 2 -i $OUT/video_frames_motion_compare/frame_%06d.png \
  -c:v libvpx-vp9 -pix_fmt yuv420p -b:v 0 -crf 30 \
  $OUT/base_e2e_lidar_motion_compare_${NUM_FRAMES}f_${TAG}.webm
```

本机 `epoch2` 已生成：

```text
UniAD/output/base_e2e_lidar_epoch2_10f/base_e2e_lidar_motion_10f_epoch2.webm
UniAD/output/base_e2e_lidar_epoch2_10f/base_e2e_lidar_motion_compare_10f_epoch2.webm
```

## 6. 已验证命令

本文档中的 `epoch2` 命令已经按顺序验证过：

- `prepare_bevformer_lidar_deploy_data.py` 生成 10 帧 raw points / metadata / GT。
- `export_bevformer_lidar_sparse_onnx.py` 导出
  `base_e2e_lidar_sparse_encoder_epoch2.onnx`。
- `export_bevformer_lidar_backbone_neck_onnx.py` 导出
  `base_e2e_lidar_backbone_neck_epoch2.onnx`。
- `export_e2e_lidar_onnx.py` 导出并 repair
  `base_e2e_lidar_trt_epoch2.repaired.onnx`。
- TensorRT 10.7 `trtexec --skipInference` 成功编译
  `base_e2e_lidar_backbone_neck_epoch2.engine` 和
  `base_e2e_lidar_trt_epoch2.engine`。
- TensorRT 10.7 `trtexec --loadEngine` 成功加载并随机输入推理 dense E2E engine。
- `uniad_lidar_e2e` 跑通 10 帧 raw-points runtime，并写出 detection、track
  state、BEV SVG 和 trajectory tensors。
- `google-chrome` + `ffmpeg` 生成 10 帧 prediction-only 和 GT-compare WebM。
- `visualize_e2e_motion.py` 生成 MotionHead 轨迹 SVG，并合成 prediction-only
  和 GT-compare motion WebM。

本机关键产物尺寸：

```text
UniAD_train/UniAD/onnx/base_e2e_lidar_sparse_encoder_epoch2.onnx      5.2M
UniAD_train/UniAD/onnx/base_e2e_lidar_backbone_neck_epoch2.onnx       17M
UniAD/onnx/base_e2e_lidar_trt_epoch2.repaired.onnx                   120M
UniAD/engine/base_e2e_lidar_backbone_neck_epoch2.engine              8.9M
UniAD/engine/base_e2e_lidar_trt_epoch2.engine                         80M
UniAD/output/base_e2e_lidar_epoch2_10f                               413M
```

## 7. 后续 checkpoint

训练继续产生新 epoch 后，推荐不要覆盖 `epoch2` 产物，改成新的 `TAG`：

```bash
export CKPT=$UNIAD_TRAIN_DIR/projects/work_dirs/stage2_e2e_lidar/base_e2e_lidar/epoch_3.pth
export TAG=epoch3
```

然后从第 2 节开始重新导出三段 ONNX、重新编译两段 engine，并用第 5 节命令跑
runtime。dense E2E engine 的 `MAX` 可以先沿用 `1201`；如果长序列上 active tracks
增长更多，再调大 profile。
