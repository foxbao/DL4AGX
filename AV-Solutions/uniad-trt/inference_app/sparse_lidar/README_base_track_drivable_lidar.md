# UniAD base_track_drivable_lidar TensorRT 部署流程

这份文档记录 `stage1_track_map_lidar/base_track_drivable_lidar.py` 的部署状态。
它基于已经跑通的 `base_track_lidar` sparse LiDAR 链路，只在 dense head 末端新增
drivable mask 输出。

```text
raw points
  -> C++ hard voxelization + HardSimpleVFE
  -> libspconv sparse encoder ONNX
  -> TensorRT LiDAR backbone+neck engine
  -> TensorRT track+drivable dense engine
  -> track states / boxes / scores / labels / drivable_score
```

## 当前状态

- 已新增部署侧 drivable head：
  `UniAD/projects/mmdet3d_plugin/uniad/dense_heads/lidar_drivable_head.py`。
- 已新增部署侧 detector：
  `UniADTrackDrivableLidarTRT`。
- 已新增部署侧 config：
  `UniAD/projects/configs/stage1_track_map_lidar/base_track_drivable_lidar_trt_p.py`。
- 已新增 dense ONNX 导出脚本：
  `UniAD/tools/export_track_drivable_lidar_onnx.py`。
- 已用 `base_track_drivable_lidar/epoch_2.pth` 完成 sparse encoder ONNX、
  backbone+neck ONNX、backbone+neck TensorRT engine、dense ONNX 和 dense
  TensorRT 10.7 engine 导出/编译。
- 已验证 TensorRT engine 可以执行随机输入 inference，最后一个输出
  `drivable_score` 的 binding shape 是 `1x120x160`。
- 已新增 C++ runtime：`inference_app/sparse_lidar/build/uniad_lidar_track_drivable`。
- 已用匹配的 sparse/front/dense 资产跑通 1 帧 raw-points C++ smoke，输出
  track txt、普通 BEV SVG、drivable score、drivable mask 和点云/track/drivable
  叠加 SVG。

当前已经生成：

```text
UniAD/onnx/base_track_drivable_lidar_trt_epoch2.onnx
UniAD/onnx/base_track_drivable_lidar_trt_epoch2.repaired.onnx
UniAD_train/UniAD/onnx/base_track_drivable_lidar_sparse_encoder_epoch2.onnx
UniAD_train/UniAD/onnx/base_track_drivable_lidar_backbone_neck_epoch2.onnx
UniAD/engine/base_track_drivable_lidar_backbone_neck_epoch2.engine
UniAD/engine/base_track_drivable_lidar_track_drivable_epoch2.engine
UniAD/dumped_inputs/base_track_drivable_lidar_trt_trace_epoch2/*.dat
UniAD_train/UniAD/dumped_inputs/base_track_drivable_lidar_sparse_encoder_epoch2/*
inference_app/sparse_lidar/build/uniad_lidar_track_drivable
inference_app/sparse_lidar/build/base_track_drivable_full_smoke/*
```

## 模型边界

`base_track_drivable_lidar.py` 继承 `base_track_lidar.py`，新增的是
`LidarDrivableHead`。部署时 dense engine 的输入仍然是：

- `lidar_bev`
- `prev_bev`
- `shift`
- timestamp / pose
- 11 个实际使用的 track state 输入
- `max_obj_id`

输出保持 `base_track_lidar` 的 21 个 track 输出不变，并在末尾追加：

```text
drivable_score: 1x120x160
```

`drivable_score` 是 float score map，C++ runtime 侧再做阈值，例如 `> 0.5`。
这样比直接导出 bool mask 更适合 TensorRT 输出和后处理可视化。

## 环境

以下命令默认从仓库根目录执行：

```bash
cd /home/baojiali/Downloads/public_code/DL4AGX/AV-Solutions/uniad-trt
conda activate uniad_train

export TRT_PATH=/home/baojiali/Downloads/TensorRT-10.7.0.23
export PATH=$TRT_PATH/bin:$PATH
export LD_LIBRARY_PATH=$TRT_PATH/lib:$LD_LIBRARY_PATH

export UNIAD_TRAIN_DIR=$PWD/UniAD_train/UniAD
export DRIVABLE_CKPT=$UNIAD_TRAIN_DIR/projects/work_dirs/stage1_track_map_lidar/base_track_drivable_lidar/epoch_2.pth
export DRIVABLE_TRT_CFG=$PWD/UniAD/projects/configs/stage1_track_map_lidar/base_track_drivable_lidar_trt_p.py
export TAG=epoch2
```

## Sparse Encoder ONNX

这一步导出 libspconv runtime 使用的 sparse encoder ONNX，同时保存一份用于检查的
libspconv tensor dump。

```bash
cd UniAD_train/UniAD

CUDA_VISIBLE_DEVICES=0 PYTHONPATH=$(pwd) \
python tools/export_bevformer_lidar_sparse_onnx.py \
  projects/configs/stage1_track_map_lidar/base_track_drivable_lidar.py \
  projects/work_dirs/stage1_track_map_lidar/base_track_drivable_lidar/epoch_2.pth \
  --split test \
  --index 0 \
  --onnx-file onnx/base_track_drivable_lidar_sparse_encoder_${TAG}.onnx \
  --tensor-prefix dumped_inputs/base_track_drivable_lidar_sparse_encoder_${TAG}/infer

cd -
```

已验证导出成功：

```text
Exported sparse ONNX: onnx/base_track_drivable_lidar_sparse_encoder_epoch2.onnx
Input features: [621662, 4], coors: [621662, 4], dense: [5, 256, 120, 160]
```

## Backbone+Neck ONNX / Engine

```bash
cd UniAD_train/UniAD

CUDA_VISIBLE_DEVICES=0 PYTHONPATH=$(pwd) \
python tools/export_bevformer_lidar_backbone_neck_onnx.py \
  projects/configs/stage1_track_map_lidar/base_track_drivable_lidar.py \
  projects/work_dirs/stage1_track_map_lidar/base_track_drivable_lidar/epoch_2.pth \
  --onnx-file onnx/base_track_drivable_lidar_backbone_neck_${TAG}.onnx

cd -
```

```bash
mkdir -p UniAD/engine

CUDA_VISIBLE_DEVICES=0 \
$TRT_PATH/bin/trtexec \
  --onnx=UniAD_train/UniAD/onnx/base_track_drivable_lidar_backbone_neck_${TAG}.onnx \
  --saveEngine=UniAD/engine/base_track_drivable_lidar_backbone_neck_${TAG}.engine \
  --fp16 \
  --skipInference
```

已验证 TensorRT 10.7.0.23 可以成功编译，engine 大约 9 MiB。

## Dense Track+Drivable ONNX

```bash
cd UniAD

CUDA_VISIBLE_DEVICES=0 PYTHONPATH=$(pwd) \
python tools/export_track_drivable_lidar_onnx.py \
  projects/configs/stage1_track_map_lidar/base_track_drivable_lidar_trt_p.py \
  ../UniAD_train/UniAD/projects/work_dirs/stage1_track_map_lidar/base_track_drivable_lidar/epoch_2.pth \
  --onnx-file ./onnx/base_track_drivable_lidar_trt_${TAG}.onnx \
  --track-state-len 601 \
  --dump-input-dir ./dumped_inputs/base_track_drivable_lidar_trt_trace_${TAG}

cd -
```

导出时的 PyTorch 输出形状应包含最后一项：

```text
(1, 120, 160)  # drivable_score
```

`onnx.checker` 会因为 `InverseTRT`、`MultiScaleDeformableAttnTRT` 等自定义
plugin op 报 unknown op，这和现有 UniAD TRT ONNX 一样；应以 TensorRT parser/
engine build 为准。

## Dense Track+Drivable Engine

```bash
mkdir -p UniAD/engine

MIN=601
OPT=601
MAX=901
TRACK_SHAPES=prev_track_intances0:Lx512,prev_track_intances1:Lx3,prev_track_intances3:L,prev_track_intances4:L,prev_track_intances5:L,prev_track_intances6:L,prev_track_intances8:L,prev_track_intances9:Lx10,prev_track_intances11:Lx4x256,prev_track_intances12:Lx4,prev_track_intances13:L

CUDA_VISIBLE_DEVICES=0 \
$TRT_PATH/bin/trtexec \
  --onnx=UniAD/onnx/base_track_drivable_lidar_trt_${TAG}.repaired.onnx \
  --saveEngine=UniAD/engine/base_track_drivable_lidar_track_drivable_${TAG}.engine \
  --staticPlugins=inference_app/enqueueV3/build/libuniad_plugin.so \
  --fp16 \
  --minShapes=${TRACK_SHAPES//L/${MIN}} \
  --optShapes=${TRACK_SHAPES//L/${OPT}} \
  --maxShapes=${TRACK_SHAPES//L/${MAX}} \
  --skipInference
```

已验证 TensorRT 10.7.0.23 可以成功编译，engine 大约 69 MiB。

## Engine Inference Smoke Test

```bash
L=601
TRACK_SHAPES=prev_track_intances0:Lx512,prev_track_intances1:Lx3,prev_track_intances3:L,prev_track_intances4:L,prev_track_intances5:L,prev_track_intances6:L,prev_track_intances8:L,prev_track_intances9:Lx10,prev_track_intances11:Lx4x256,prev_track_intances12:Lx4,prev_track_intances13:L

CUDA_VISIBLE_DEVICES=0 \
$TRT_PATH/bin/trtexec \
  --loadEngine=UniAD/engine/base_track_drivable_lidar_track_drivable_${TAG}.engine \
  --staticPlugins=inference_app/enqueueV3/build/libuniad_plugin.so \
  --shapes=${TRACK_SHAPES//L/${L}} \
  --duration=1 \
  --warmUp=0
```

已验证输出 binding 包含：

```text
drivable_score: 1x120x160
```

随机输入 smoke 的平均 Host latency 约 15 ms；这个数只用于确认 engine 能跑，
不代表真实端到端性能。

## C++ Runtime

已新增一个独立二进制，避免影响已经验证过的 `uniad_lidar_track`：

```text
uniad_lidar_track_drivable
```

实现方式：

- 复用 `uniad_lidar_track` 的 sparse encoder、backbone+neck、track state 管理。
- dense engine 使用 `base_track_drivable_lidar_track_drivable_${TAG}.engine`。
- 读取新增输出 `drivable_score`，shape 为 `1x120x160`。
- 将 `drivable_score` 保存为 `frame_xxxxxx_drivable_score.bin`。
- 阈值化得到 drivable mask，保存为 `frame_xxxxxx_drivable_mask.pgm`。
- 可视化保存为 `frame_xxxxxx_drivable.svg`，包含点云、track boxes 和半透明
  drivable mask。
- `drivable.svg` 的 mask 坐标按训练真值生成约定绘制：`col -> x`，
  `row -> y`。

- sparse encoder ONNX 使用 `base_track_drivable_lidar.py` 和对应 checkpoint
  导出。
- backbone+neck engine 使用同一个 checkpoint 导出的 ONNX 编译。

### Build

```bash
cmake --build inference_app/sparse_lidar/build -j$(nproc)
```

成功后应有：

```text
inference_app/sparse_lidar/build/uniad_lidar_track
inference_app/sparse_lidar/build/uniad_lidar_track_drivable
```

### Raw-Points Smoke

```bash
rm -rf inference_app/sparse_lidar/build/base_track_drivable_full_smoke
mkdir -p inference_app/sparse_lidar/build/base_track_drivable_full_smoke

CUDA_VISIBLE_DEVICES=0 \
inference_app/sparse_lidar/build/uniad_lidar_track_drivable \
  UniAD_train/UniAD/onnx/base_track_drivable_lidar_sparse_encoder_epoch2.onnx \
  UniAD/engine/base_track_drivable_lidar_backbone_neck_epoch2.engine \
  UniAD/engine/base_track_drivable_lidar_track_drivable_epoch2.engine \
  inference_app/enqueueV3/build/libuniad_plugin.so \
  UniAD_train/UniAD/dumped_inputs/base_track_lidar_deploy_data_10f \
  inference_app/sparse_lidar/build/base_track_drivable_full_smoke \
  1 \
  --metadata-json UniAD_train/UniAD/dumped_inputs/base_track_lidar_deploy_data_10f \
  --track-init-dir UniAD/dumped_inputs/base_track_drivable_lidar_trt_trace_epoch2 \
  --bev-max-points 15000 \
  --bev-score-threshold 0.25 \
  --drivable-threshold 0.5
```

已验证生成：

```text
frame_000000_detections.txt
frame_000000_bev.svg
frame_000000_drivable_score.bin   # 120*160 float32
frame_000000_drivable_mask.pgm    # 160x120
frame_000000_drivable.svg         # point cloud + track + drivable overlay
```

本次 smoke 的关键日志：

```text
track engine exposes drivable_score; drivable outputs will be written.
generated sparse input: 124329 voxels, max_points_per_voxel=10, max_voxels=160000
LiDAR Backbone+Neck TensorRT output lidar_bev: 1 x 256 x 120 x 160
TRT output drivable_score: run=1 x 120 x 160 alloc=1 x 120 x 160 dtype=float32
frame 0 outputs written ... (4 tracks, max_obj_id=4)
```
