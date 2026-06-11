# UniAD LiDAR Sparse Encoder Runtime

This directory contains the C++ LiDAR runtime pieces for the exported
`base_bevformer_lidar.py` flow. The current path is:

```text
raw points
  -> C++ hard voxelization + HardSimpleVFE mean feature
  -> libspconv sparse ONNX
  -> TensorRT LiDAR backbone+neck
  -> TensorRT dense-BEV head
  -> raw tensors, decoded detections, BEV SVG
```

The sparse ONNX is loaded by libspconv directly. Only the LiDAR backbone+neck
and dense-BEV head are TensorRT engines.

## Environment

Python tools use the training environment; TensorRT tools and the C++ runtime
use the TensorRT 10.7 runtime in this workspace.

```bash
conda activate uniad_train

export TRT_PATH=/home/baojiali/Downloads/TensorRT-10.7.0.23
export PATH=$TRT_PATH/bin:$PATH
export LD_LIBRARY_PATH=$TRT_PATH/lib:$LD_LIBRARY_PATH
```

TensorRT engine plans are minor-version sensitive. The epoch-2 engines in this
workspace were built and smoke-tested with TensorRT 10.7.

## Build

```bash
cmake -S inference_app/sparse_lidar -B inference_app/sparse_lidar/build \
  -DLIDAR_AI_SOLUTION_PATH=/home/baojiali/Downloads/public_code/Lidar_AI_Solution \
  -DSPCONV_CUDA_VERSION=11.4 \
  -DTENSORRT_PATH=/home/baojiali/Downloads/TensorRT-10.7.0.23
cmake --build inference_app/sparse_lidar/build -j$(nproc)
```

`CUDA_TOOLKIT_ROOT_DIR` defaults to `$CUDA_HOME`, `$CUDA_PATH`, then
`/usr/local/cuda`.

## Prepare Deployment Input Data

This is the deployment data step. It only builds the configured dataset and
writes raw points, metadata, and optional GT boxes. It does not load a model,
does not load a checkpoint, and does not run PyTorch forward.

```bash
cd UniAD_train/UniAD
conda activate uniad_train

python tools/prepare_bevformer_lidar_deploy_data.py \
  --config projects/configs/bevformer_lidar/base_bevformer_lidar.py \
  --split test \
  --start-index 0 \
  --max-frames 1 \
  --out-dir dumped_inputs/bevformer_lidar_deploy_data

cd -
```

The output directory contains:

```text
raw_points_000000.bin       # deployment input points, float32 Nx4
raw_points_000000.npy       # same data for inspection/debug
img_metas_000000.json       # metadata used for dense-BEV shift
gt_detections_000000.txt    # optional visualization reference
manifest.json               # config, split, token, scene, and file list
```

Use `--no-gt` for a pure deployment data package without GT visualization files.

## Run Deployment

```bash
export TRT_PATH=/home/baojiali/Downloads/TensorRT-10.7.0.23
export PATH=$TRT_PATH/bin:$PATH
export LD_LIBRARY_PATH=$TRT_PATH/lib:$LD_LIBRARY_PATH

cmake --build inference_app/sparse_lidar/build -j$(nproc)

./inference_app/sparse_lidar/build/uniad_lidar \
  UniAD_train/UniAD/onnx/bevformer_lidar_sparse_encoder_epoch2.onnx \
  UniAD/engine/bevformer_lidar_backbone_neck_epoch2_trt10.7_sm89.engine \
  UniAD/engine/bevformer_lidar_bev_trt_epoch2_trt10.7_sm89.engine \
  inference_app/enqueueV3/build_trt107/libuniad_plugin.so \
  UniAD_train/UniAD/dumped_inputs/bevformer_lidar_deploy_data \
  inference_app/sparse_lidar/build/uniad_lidar_epoch2_deploy_data \
  1 \
  41 960 1280 \
  --metadata-json UniAD_train/UniAD/dumped_inputs/bevformer_lidar_deploy_data \
  --gt-detections UniAD_train/UniAD/dumped_inputs/bevformer_lidar_deploy_data \
  --score-threshold 0.05 \
  --bev-score-threshold 0.05
```

For deployment without GT comparison, omit `--gt-detections`. Outputs are
written per frame with a `frame_XXXXXX` prefix:

```text
frame_000000_lidar_bev.bin
frame_000000_bev_embed.bin
frame_000000_all_cls_scores.bin
frame_000000_all_bbox_preds.bin
frame_000000_detections.txt
frame_000000_bev.svg
frame_000000_bev_compare.svg    # only when --gt-detections is provided
```

`frame_XXXXXX_detections.txt` is decoded with the same NMS-free logic as
`NMSFreeCoder`: last decoder layer, sigmoid class scores, top-300 by default,
box size `exp`, yaw `atan2(sin, cos)`, then `post_center_range` filtering.
Use `--score-threshold <score>` to filter decoded detections and
`--max-dets <count>` to change the top-k count. The BEV SVG draws the first
100 decoded boxes by default; use `--bev-max-draw <count>` and
`--bev-score-threshold <score>` to control visualization clutter.

The raw-points input can be a single file when `num_frames=1`, a directory
with files such as `raw_points_000000.bin`, or a printf-style pattern such as
`/data/points/raw_points_%06d.bin`.

Pass `--metadata-json <file_or_dir_or_pattern>` to compute the dense-BEV
`shift` from UniAD metadata instead of feeding dumped shift binaries. The
reader selects the current item from `queue_metas`, uses `ego_motion_delta`,
and honors `prev_bev_exists=false` by resetting the C++ `prev_bev` state and
feeding `use_prev_bev=0`. If both `--metadata-json` and `--shift-dir` are
provided, metadata-derived shift takes precedence.

## Optional PyTorch Golden Reference

Run this only when you need PyTorch-vs-TensorRT numerical comparison or
visualization. This path loads the checkpoint and dumps PyTorch model tensors;
it is intentionally separate from deployment input data preparation.

```bash
cd UniAD_train/UniAD
conda activate uniad_train

python tools/dump_bevformer_lidar_golden.py \
  projects/configs/bevformer_lidar/base_bevformer_lidar.py \
  projects/work_dirs/bevformer_lidar/base_bevformer_lidar/epoch_2.pth \
  --split test \
  --index 0 \
  --out-dir dumped_inputs/bevformer_lidar_raw_golden_epoch2

cd -
```

## Sparse Encoder Check

```bash
./inference_app/sparse_lidar/build/validate_sparse \
  UniAD_train/UniAD/onnx/bevformer_lidar_sparse_encoder_epoch2.onnx \
  UniAD_train/UniAD/dumped_inputs/bevformer_lidar_sparse_encoder_epoch2/infer.voxels \
  UniAD_train/UniAD/dumped_inputs/bevformer_lidar_sparse_encoder_epoch2/infer.coors \
  inference_app/sparse_lidar/build/bevformer_lidar_sparse_output.dense \
  UniAD_train/UniAD/dumped_inputs/bevformer_lidar_sparse_encoder_epoch2/infer.dense \
  41 960 1280
```

Set `SPARSE_LIDAR_VERBOSE=1` to enable libspconv layer logs.

## Sparse + TRT Check

The sparse ONNX exports `middle_bev`. Run the LiDAR backbone+neck TensorRT
engine before feeding the dense BEVFormer TensorRT engine.

Export and build the backbone+neck engine. The `--golden-dir` argument is
optional; include it when you want the export script to compare against the
PyTorch golden dump from the previous section.

```bash
cd UniAD_train/UniAD
conda activate uniad_train

python tools/export_bevformer_lidar_backbone_neck_onnx.py \
  projects/configs/bevformer_lidar/base_bevformer_lidar.py \
  projects/work_dirs/bevformer_lidar/base_bevformer_lidar/epoch_2.pth \
  --golden-dir dumped_inputs/bevformer_lidar_raw_golden_epoch2/test_000000 \
  --onnx-file onnx/bevformer_lidar_backbone_neck_epoch2.onnx
cd -

trtexec \
  --onnx=UniAD_train/UniAD/onnx/bevformer_lidar_backbone_neck_epoch2.onnx \
  --saveEngine=UniAD/engine/bevformer_lidar_backbone_neck_epoch2_trt10.7_sm89.engine \
  --fp16 --skipInference
```

```bash
./inference_app/sparse_lidar/build/validate_sparse_trt \
  UniAD_train/UniAD/onnx/bevformer_lidar_sparse_encoder_epoch2.onnx \
  UniAD_train/UniAD/dumped_inputs/bevformer_lidar_sparse_encoder_epoch2/infer.voxels \
  UniAD_train/UniAD/dumped_inputs/bevformer_lidar_sparse_encoder_epoch2/infer.coors \
  UniAD/engine/bevformer_lidar_backbone_neck_epoch2_trt10.7_sm89.engine \
  UniAD/engine/bevformer_lidar_bev_trt_epoch2_trt10.7_sm89.engine \
  inference_app/enqueueV3/build_trt107/libuniad_plugin.so \
  UniAD_train/UniAD/dumped_inputs/bevformer_lidar_raw_golden_epoch2/test_000000 \
  inference_app/sparse_lidar/build/sparse_frontend_dense_trt_epoch2 \
  41 960 1280
```

On the TensorRT 10.7 / SM89 engines in this workspace, the raw-points
validation produced 424653 input points and 123922 voxels. The dense outputs
matched the PyTorch golden dump with:

```text
lidar_bev:       max_abs 0.070555687, mean_abs 0.001351151, p99_abs 0.011542119, cosine 0.999991748
bev_embed:       max_abs 0.251663744, mean_abs 0.006279593, p99_abs 0.028822303, cosine 0.999958906
all_cls_scores:  max_abs 0.180603504, mean_abs 0.003920786, p99_abs 0.027340889, cosine 0.999999430
all_bbox_preds:  max_abs 0.652954102, mean_abs 0.009555208, p99_abs 0.122215271, cosine 0.999998209
```

The metadata-derived shift matches the exported golden `dense/shift.bin` for
the dumped raw-points sample. Full multi-frame parity with the Python dense
path still requires the Python `rotate_prev_bev_if_needed()` operation to be
implemented in C++/CUDA or exported into the runtime; this CLI currently feeds
the previous BEV state without that external rotation.

If neither `--metadata-json` nor `--shift-dir` is provided, the CLI uses zero
shift for every frame and prints a warning. To feed precomputed frame shifts,
pass a directory containing files such as `shift_0.bin`, `shift_000000.bin`,
or `000000/shift.bin`.

To validate the raw-points runtime front-end, replace the dumped sparse inputs
with `--raw-points`. The C++ front-end hard-voxelizes `Nx4` float32 points with
the `base_bevformer_lidar.py` voxel settings, applies the HardSimpleVFE mean
feature, then feeds libspconv.

```bash
./inference_app/sparse_lidar/build/validate_sparse_trt \
  UniAD_train/UniAD/onnx/bevformer_lidar_sparse_encoder_epoch2.onnx \
  --raw-points \
  UniAD_train/UniAD/dumped_inputs/bevformer_lidar_raw_golden_epoch2/test_000000/current/raw_points_0.bin \
  UniAD/engine/bevformer_lidar_backbone_neck_epoch2_trt10.7_sm89.engine \
  UniAD/engine/bevformer_lidar_bev_trt_epoch2_trt10.7_sm89.engine \
  inference_app/enqueueV3/build_trt107/libuniad_plugin.so \
  UniAD_train/UniAD/dumped_inputs/bevformer_lidar_raw_golden_epoch2/test_000000 \
  inference_app/sparse_lidar/build/raw_points_frontend_dense_trt_epoch2 \
  41 960 1280
```
