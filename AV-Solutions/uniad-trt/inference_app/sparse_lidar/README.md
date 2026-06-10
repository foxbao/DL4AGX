# UniAD LiDAR Sparse Encoder Runtime Check

This sample is a minimal C++ validation harness for the libspconv sparse
encoder exported from `UniAD_train/UniAD`.

It loads the custom sparse ONNX graph, feeds dumped voxel features and
coordinates, writes the dense BEV tensor, and optionally compares it against a
reference tensor.

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

## Sparse Encoder Check

```bash
./inference_app/sparse_lidar/build/validate_sparse \
  UniAD_train/UniAD/onnx/bevformer_lidar_sparse_encoder_epoch1.onnx \
  UniAD_train/UniAD/dumped_inputs/bevformer_lidar_sparse_encoder_epoch1/infer.voxels \
  UniAD_train/UniAD/dumped_inputs/bevformer_lidar_sparse_encoder_epoch1/infer.coors \
  inference_app/sparse_lidar/build/bevformer_lidar_sparse_output.dense \
  UniAD_train/UniAD/dumped_inputs/bevformer_lidar_sparse_encoder_epoch1/infer.dense \
  41 960 1280
```

Set `SPARSE_LIDAR_VERBOSE=1` to enable libspconv layer logs.

## Sparse + TRT Check

The sparse ONNX exports `middle_bev`. Run the LiDAR backbone+neck TensorRT
engine before feeding the dense BEVFormer TensorRT engine.

Export and build the backbone+neck engine:

```bash
cd UniAD_train/UniAD
/home/baojiali/anaconda3/envs/uniad_train/bin/python \
  tools/export_bevformer_lidar_backbone_neck_onnx.py \
  projects/configs/bevformer_lidar/base_bevformer_lidar.py \
  projects/work_dirs/bevformer_lidar/base_bevformer_lidar/epoch_1.pth \
  --golden-dir dumped_inputs/bevformer_lidar_raw_golden/test_000000 \
  --onnx-file onnx/bevformer_lidar_backbone_neck_epoch1.onnx
cd -
LD_LIBRARY_PATH=/home/baojiali/Downloads/TensorRT-10.7.0.23/targets/x86_64-linux-gnu/lib:$LD_LIBRARY_PATH \
  /home/baojiali/Downloads/TensorRT-10.7.0.23/targets/x86_64-linux-gnu/bin/trtexec \
  --onnx=UniAD_train/UniAD/onnx/bevformer_lidar_backbone_neck_epoch1.onnx \
  --saveEngine=UniAD/engine/bevformer_lidar_backbone_neck_epoch1_trt10.7_sm89.engine \
  --fp16 --skipInference
```

```bash
./inference_app/sparse_lidar/build/validate_sparse_trt \
  UniAD_train/UniAD/onnx/bevformer_lidar_sparse_encoder_epoch1.onnx \
  UniAD_train/UniAD/dumped_inputs/bevformer_lidar_sparse_encoder_epoch1/infer.voxels \
  UniAD_train/UniAD/dumped_inputs/bevformer_lidar_sparse_encoder_epoch1/infer.coors \
  UniAD/engine/bevformer_lidar_backbone_neck_epoch1_trt10.7_sm89.engine \
  UniAD/engine/bevformer_lidar_bev_trt_epoch1_trt10.7_sm89.engine \
  inference_app/enqueueV3/build_trt107/libuniad_plugin.so \
  UniAD_train/UniAD/dumped_inputs/bevformer_lidar_raw_golden/test_000000 \
  inference_app/sparse_lidar/build/sparse_frontend_dense_trt \
  41 960 1280
```
