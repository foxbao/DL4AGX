# UniAD LiDAR TensorRT 部署流程

这份文档记录 `base_bevformer_lidar.py` 的 LiDAR 部署链路。当前链路是：

```text
raw points
  -> C++ hard voxelization + HardSimpleVFE
  -> libspconv sparse ONNX
  -> TensorRT LiDAR backbone+neck engine
  -> TensorRT Dense-BEV head engine
  -> detections / BEV SVG / GT-vs-Pred SVG
```

注意：sparse encoder 的 ONNX 由 `libspconv` 直接加载，不编译成 TensorRT
engine。只有 `backbone+neck` 和 `Dense-BEV head` 两段会生成 TensorRT engine。

## 命名约定

- `UniAD_train/UniAD`：训练侧代码，用来导出 sparse encoder ONNX、
  backbone+neck ONNX、准备部署输入数据、dump PyTorch golden。
- `UniAD`：部署侧代码，用来导出 Dense-BEV TensorRT 边界 ONNX。
- `inference_app/enqueueV3`：TensorRT 10.x 的 plugin 工程，生成
  `libuniad_plugin.so`。
- `inference_app/sparse_lidar`：LiDAR C++ runtime 和验证工具。
- `dependencies/3DSparseConvolution`：repo 内置的 `libspconv` C++ runtime
  最小集合，`sparse_lidar` 默认会自动使用它。
- `dumped_inputs/bevformer_lidar_deploy_data` 不带 `epoch`，因为它只是数据
  和 metadata，不依赖 checkpoint。
- README 默认使用稳定产物名，例如 `bevformer_lidar_sparse_encoder.onnx`、
  `bevformer_lidar_backbone_neck.engine`。如果要同时保留多个 checkpoint 的
  产物，可以手动在文件名后加 tag，例如 `_best_20260611` 或 `_run042`。

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
# 改成当前要部署的 checkpoint，例如 epoch_2.pth、latest.pth 或 best.pth。
export CKPT=$UNIAD_TRAIN_DIR/projects/work_dirs/bevformer_lidar/base_bevformer_lidar/epoch_2.pth
```

这里不设置 `CUDA_VISIBLE_DEVICES`。如果要换 GPU，用系统默认 CUDA 选择方式
或在外层运行环境里处理。

如果需要使用别处的 `3DSparseConvolution`，可以额外传
`-DSPARSE_CONV_ROOT=/path/to/3DSparseConvolution`。旧的
`-DLIDAR_AI_SOLUTION_PATH=/path/to/Lidar_AI_Solution` 仍然兼容，但不推荐作为
长期命令。

## 1. 准备部署输入数据

这一步只读 dataset，写部署 runtime 需要的 raw points、metadata 和可选 GT。
它不加载 checkpoint，也不跑 PyTorch forward。

```bash
cd UniAD_train/UniAD

python tools/prepare_bevformer_lidar_deploy_data.py \
  --config projects/configs/bevformer_lidar/base_bevformer_lidar.py \
  --split test \
  --start-index 0 \
  --max-frames 1 \
  --out-dir dumped_inputs/bevformer_lidar_deploy_data

cd -
```

输出目录示例：

```text
raw_points_000000.bin       # float32 Nx4，部署输入
raw_points_000000.npy       # 同一份点云，方便 debug
img_metas_000000.json       # Dense-BEV shift / prev_bev_exists metadata
gt_detections_000000.txt    # 可选，用于左右对比图
manifest.json               # config、split、token、scene、文件列表
```

如果只需要纯部署输入，不需要画 GT 对比，加 `--no-gt`。

## 2. 导出 ONNX

下面以 `CKPT` 指向当前要部署的 checkpoint。换 checkpoint 时，只需要更新
`CKPT`；如果不需要保留旧产物，ONNX 和 engine 文件名可以保持不变并覆盖。

### 2.1 Sparse Encoder ONNX

```bash
cd UniAD_train/UniAD

python tools/export_bevformer_lidar_sparse_onnx.py \
  projects/configs/bevformer_lidar/base_bevformer_lidar.py \
  $CKPT \
  --split test \
  --index 0 \
  --onnx-file onnx/bevformer_lidar_sparse_encoder.onnx \
  --tensor-prefix dumped_inputs/bevformer_lidar_sparse_encoder/infer

cd -
```

这一步会生成：

```text
UniAD_train/UniAD/onnx/bevformer_lidar_sparse_encoder.onnx
UniAD_train/UniAD/dumped_inputs/bevformer_lidar_sparse_encoder/infer.voxels
UniAD_train/UniAD/dumped_inputs/bevformer_lidar_sparse_encoder/infer.coors
UniAD_train/UniAD/dumped_inputs/bevformer_lidar_sparse_encoder/infer.dense
```

`infer.*` 主要用于验证 sparse ONNX，本身不是正式部署输入数据。

### 2.2 LiDAR Backbone+Neck ONNX

```bash
cd UniAD_train/UniAD

python tools/export_bevformer_lidar_backbone_neck_onnx.py \
  projects/configs/bevformer_lidar/base_bevformer_lidar.py \
  $CKPT \
  --onnx-file onnx/bevformer_lidar_backbone_neck.onnx

cd -
```

如果已经 dump 了 PyTorch golden，并且想在导出时顺手比较，可额外加：

```bash
--golden-dir dumped_inputs/bevformer_lidar_raw_golden/test_000000
```

### 2.3 Dense-BEV Head ONNX

Dense-BEV head 使用部署侧 `UniAD` 里的 TensorRT plugin symbolic 路径导出。
脚本会同时写原始 ONNX 和修复过 `Reshape.allowzero` 的 `.repaired.onnx`。
TensorRT 实际编译时使用 `.repaired.onnx`。

```bash
cd UniAD

python tools/export_bevformer_lidar_onnx.py \
  projects/configs/bevformer_lidar/base_bevformer_lidar_trt_p.py \
  $CKPT \
  --onnx-file onnx/bevformer_lidar_bev_trt.onnx

cd -
```

输出：

```text
UniAD/onnx/bevformer_lidar_bev_trt.onnx
UniAD/onnx/bevformer_lidar_bev_trt.repaired.onnx
```

## 3. 编译 C++ 和 TensorRT Plugin

### 3.1 编译 TensorRT plugin

```bash
cmake -S inference_app/enqueueV3 -B inference_app/enqueueV3/build_trt107 \
  -DTENSORRT_PATH=$TRT_PATH \
  -DTARGET_GPU_SM=$TARGET_GPU_SM

cmake --build inference_app/enqueueV3/build_trt107 -j$(nproc)
```

输出：

```text
inference_app/enqueueV3/build_trt107/libuniad_plugin.so
```

### 3.2 编译 sparse_lidar runtime

```bash
cmake -S inference_app/sparse_lidar -B inference_app/sparse_lidar/build \
  -DSPCONV_CUDA_VERSION=$SPCONV_CUDA_VERSION \
  -DTENSORRT_PATH=$TRT_PATH

cmake --build inference_app/sparse_lidar/build -j$(nproc)
```

输出包括：

```text
inference_app/sparse_lidar/build/validate_sparse
inference_app/sparse_lidar/build/validate_sparse_trt
inference_app/sparse_lidar/build/uniad_lidar
```

## 4. 编译 TensorRT Engine

先创建 engine 目录：

```bash
mkdir -p UniAD/engine
```

### 4.1 Backbone+Neck Engine

```bash
$TRT_PATH/bin/trtexec \
  --onnx=UniAD_train/UniAD/onnx/bevformer_lidar_backbone_neck.onnx \
  --saveEngine=UniAD/engine/bevformer_lidar_backbone_neck_trt10.7_sm89.engine \
  --fp16 \
  --skipInference
```

### 4.2 Dense-BEV Head Engine

Dense-BEV ONNX 里有 TensorRT plugin op，所以编译时要加载
`libuniad_plugin.so`。

```bash
$TRT_PATH/bin/trtexec \
  --onnx=UniAD/onnx/bevformer_lidar_bev_trt.repaired.onnx \
  --saveEngine=UniAD/engine/bevformer_lidar_bev_head_trt10.7_sm89.engine \
  --staticPlugins=inference_app/enqueueV3/build_trt107/libuniad_plugin.so \
  --fp16 \
  --skipInference
```

## 5. 运行部署链

`uniad_lidar` 会从 raw points 开始跑完整链路，并在多帧时把上一帧
`bev_embed` 作为下一帧 `prev_bev`。第 0 帧会使用零 `prev_bev` 和
`use_prev_bev=0`；metadata 里的 `prev_bev_exists=false` 会清空 C++ 侧
prev-BEV 状态。

```bash
./inference_app/sparse_lidar/build/uniad_lidar \
  UniAD_train/UniAD/onnx/bevformer_lidar_sparse_encoder.onnx \
  UniAD/engine/bevformer_lidar_backbone_neck_trt10.7_sm89.engine \
  UniAD/engine/bevformer_lidar_bev_head_trt10.7_sm89.engine \
  inference_app/enqueueV3/build_trt107/libuniad_plugin.so \
  UniAD_train/UniAD/dumped_inputs/bevformer_lidar_deploy_data \
  inference_app/sparse_lidar/build/uniad_lidar_deploy_data \
  1 \
  41 960 1280 \
  --metadata-json UniAD_train/UniAD/dumped_inputs/bevformer_lidar_deploy_data \
  --gt-detections UniAD_train/UniAD/dumped_inputs/bevformer_lidar_deploy_data \
  --score-threshold 0.05 \
  --bev-score-threshold 0.05
```

如果不需要 GT / prediction 左右对比图，去掉 `--gt-detections`。

输出文件按 frame 前缀写入：

```text
frame_000000_lidar_bev.bin
frame_000000_bev_embed.bin
frame_000000_all_cls_scores.bin
frame_000000_all_bbox_preds.bin
frame_000000_detections.txt
frame_000000_bev.svg
frame_000000_bev_compare.svg    # 只有传了 --gt-detections 才会生成
```

`frame_XXXXXX_detections.txt` 的 decode 逻辑对齐 `NMSFreeCoder`：取最后一层
decoder，class score 做 sigmoid，默认 top-300，box size 用 `exp`，yaw 用
`atan2(sin, cos)`，最后做 `post_center_range` 过滤。

常用过滤参数：

```text
--score-threshold <score>      # detections.txt 的分数过滤
--max-dets <count>             # top-k 数量，默认 300
--bev-score-threshold <score>  # BEV SVG 的分数过滤
--bev-max-draw <count>         # BEV SVG 最多画多少个框，默认 100
```

## 6. 可选：PyTorch Golden 和数值对照

这一节只在需要 PyTorch-vs-TensorRT 数值对照或可视化对照时运行。正式部署
不需要先 dump golden。

### 6.1 Dump PyTorch Golden

```bash
cd UniAD_train/UniAD

python tools/dump_bevformer_lidar_golden.py \
  projects/configs/bevformer_lidar/base_bevformer_lidar.py \
  $CKPT \
  --split test \
  --index 0 \
  --out-dir dumped_inputs/bevformer_lidar_raw_golden

cd -
```

### 6.2 单独检查 Sparse ONNX

```bash
./inference_app/sparse_lidar/build/validate_sparse \
  UniAD_train/UniAD/onnx/bevformer_lidar_sparse_encoder.onnx \
  UniAD_train/UniAD/dumped_inputs/bevformer_lidar_sparse_encoder/infer.voxels \
  UniAD_train/UniAD/dumped_inputs/bevformer_lidar_sparse_encoder/infer.coors \
  inference_app/sparse_lidar/build/bevformer_lidar_sparse_output.dense \
  UniAD_train/UniAD/dumped_inputs/bevformer_lidar_sparse_encoder/infer.dense \
  41 960 1280
```

如果需要 libspconv layer log：

```bash
SPARSE_LIDAR_VERBOSE=1 ./inference_app/sparse_lidar/build/validate_sparse ...
```

### 6.3 Raw Points 到 Dense-BEV 的端到端数值对照

```bash
./inference_app/sparse_lidar/build/validate_sparse_trt \
  UniAD_train/UniAD/onnx/bevformer_lidar_sparse_encoder.onnx \
  --raw-points \
  UniAD_train/UniAD/dumped_inputs/bevformer_lidar_raw_golden/test_000000/current/raw_points_0.bin \
  UniAD/engine/bevformer_lidar_backbone_neck_trt10.7_sm89.engine \
  UniAD/engine/bevformer_lidar_bev_head_trt10.7_sm89.engine \
  inference_app/enqueueV3/build_trt107/libuniad_plugin.so \
  UniAD_train/UniAD/dumped_inputs/bevformer_lidar_raw_golden/test_000000 \
  inference_app/sparse_lidar/build/raw_points_frontend_dense_trt \
  41 960 1280
```

本地 TensorRT 10.7 / SM89 engine 的一次结果：

```text
raw points: 424653
voxels:     123922

lidar_bev:       max_abs 0.070555687, mean_abs 0.001351151, p99_abs 0.011542119, cosine 0.999991748
bev_embed:       max_abs 0.251663744, mean_abs 0.006279593, p99_abs 0.028822303, cosine 0.999958906
all_cls_scores:  max_abs 0.180603504, mean_abs 0.003920786, p99_abs 0.027340889, cosine 0.999999430
all_bbox_preds:  max_abs 0.652954102, mean_abs 0.009555208, p99_abs 0.122215271, cosine 0.999998209
```

### 6.4 PyTorch BEV 可视化

```bash
cd UniAD_train/UniAD

python tools/visualize_bevformer_lidar_pytorch.py \
  --config projects/configs/bevformer_lidar/base_bevformer_lidar.py \
  --checkpoint $CKPT \
  --split test \
  --start-index 0 \
  --max-frames 1 \
  --score-thr 0.05 \
  --out-dir projects/work_dirs/vis_bevformer_lidar_pytorch \
  --annotate

cd -
```

这个工具输出 PNG 和 `index.html`，左边画 GT，右边画 PyTorch 推理结果。

## 7. 运行时输入格式

`uniad_lidar` 的 raw-points 输入支持三种形式：

```text
单文件:  raw_points.bin                     # num_frames 必须是 1
目录:    dumped_inputs/bevformer_lidar_deploy_data
pattern: /data/points/raw_points_%06d.bin
```

目录模式会自动查找：

```text
raw_points_0.bin
raw_points_000000.bin
000000.bin
000000/raw_points_0.bin
000000/current/raw_points_0.bin
test_000000/current/raw_points_0.bin
```

`--metadata-json` 和 `--gt-detections` 也支持单文件、目录或 pattern。当前
`prepare_bevformer_lidar_deploy_data.py` 生成的
`img_metas_000000.json`、`gt_detections_000000.txt` 可以被 runtime 直接识别。

## 8. 当前限制

- 多帧时，C++ runtime 会传递上一帧 `bev_embed`，但还没有实现 Python 里的
  `rotate_prev_bev_if_needed()`。如果要做严格多帧数值对齐，需要把这一步也
  搬到 C++/CUDA 或导出进 runtime。
- TensorRT engine 和 TensorRT minor version 绑定很强。这里的 engine 以
  TensorRT 10.7 生成和验证，运行时也应使用 TensorRT 10.7 的库。
- `bevformer_lidar_deploy_data` 是数据目录，可以复用；ONNX、engine、golden
  是权重相关产物，换 checkpoint 后需要重新生成。
