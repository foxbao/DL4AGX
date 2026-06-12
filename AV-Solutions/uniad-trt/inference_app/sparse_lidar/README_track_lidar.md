# UniAD base_track_lidar TensorRT 部署流程

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
- 已用一轮训练 checkpoint 导出并编译：
  `UniAD_train/UniAD/projects/work_dirs/stage1_track_map_lidar/base_track_lidar/epoch_1.pth`。
- 已生成：
  `UniAD_train/UniAD/onnx/base_track_lidar_sparse_encoder.onnx`、
  `UniAD_train/UniAD/onnx/base_track_lidar_backbone_neck.onnx`、
  `UniAD/onnx/base_track_lidar_trt.repaired.onnx`、
  `UniAD/engine/base_track_lidar_backbone_neck.engine`、
  `UniAD/engine/base_track_lidar_track_head.engine`。
- 已新增 C++ runtime：`inference_app/sparse_lidar/build/uniad_lidar_track`。
- 已用 `epoch_1.pth` 跑通 2 帧 raw-points smoke test，输出 track state、
  detection txt 和 BEV SVG。

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
export NUM_FRAMES=2
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
  --out-dir dumped_inputs/base_track_lidar_deploy_data

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
  --onnx-file onnx/base_track_lidar_sparse_encoder.onnx \
  --tensor-prefix dumped_inputs/base_track_lidar_sparse_encoder/infer

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
  --onnx-file onnx/base_track_lidar_backbone_neck.onnx

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
  --onnx-file onnx/base_track_lidar_trt.onnx \
  --track-state-len 601 \
  --dump-input-dir dumped_inputs/base_track_lidar_trt_trace

cd -
```

输出：

```text
UniAD/onnx/base_track_lidar_trt.onnx
UniAD/onnx/base_track_lidar_trt.repaired.onnx
UniAD/dumped_inputs/base_track_lidar_trt_trace/*.dat
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

```bash
mkdir -p UniAD/engine
```

### 4.1 Backbone+Neck Engine

```bash
$TRT_PATH/bin/trtexec \
  --onnx=UniAD_train/UniAD/onnx/base_track_lidar_backbone_neck.onnx \
  --saveEngine=UniAD/engine/base_track_lidar_backbone_neck.engine \
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
  --onnx=UniAD/onnx/base_track_lidar_trt.repaired.onnx \
  --saveEngine=UniAD/engine/base_track_lidar_track_head.engine \
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

首帧和 scene reset 会从 `UniAD/dumped_inputs/base_track_lidar_trt_trace`
恢复初始 state。这个目录由 dense track ONNX export 生成，不能随便用全零替代，
因为 `prev_track_intances0/1` 里包含 learned query embedding 和 reference points。

```bash
export TRT_PATH=/home/baojiali/Downloads/TensorRT-10.7.0.23
export LD_LIBRARY_PATH=$TRT_PATH/lib:$LD_LIBRARY_PATH
export CUDA_VISIBLE_DEVICES=0

rm -rf UniAD/output/base_track_lidar_track_epoch1

inference_app/sparse_lidar/build/uniad_lidar_track \
  UniAD_train/UniAD/onnx/base_track_lidar_sparse_encoder.onnx \
  UniAD/engine/base_track_lidar_backbone_neck.engine \
  UniAD/engine/base_track_lidar_track_head.engine \
  inference_app/enqueueV3/build/libuniad_plugin.so \
  UniAD_train/UniAD/dumped_inputs/base_track_lidar_deploy_data \
  UniAD/output/base_track_lidar_track_epoch1 \
  2 \
  --metadata-json UniAD_train/UniAD/dumped_inputs/base_track_lidar_deploy_data \
  --gt-detections UniAD_train/UniAD/dumped_inputs/base_track_lidar_deploy_data \
  --track-init-dir UniAD/dumped_inputs/base_track_lidar_trt_trace \
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

`epoch_1.pth` 的 2 帧 smoke test 当前输出：

```text
frame 0: 4 tracks, max_obj_id=4
frame 1: 4 tracks, max_obj_id=4
```

输出目录：

```text
UniAD/output/base_track_lidar_track_epoch1
```

## 6. 已验证命令

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
