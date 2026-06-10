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
  -DSPCONV_CUDA_VERSION=11.4
cmake --build inference_app/sparse_lidar/build -j$(nproc)
```

`CUDA_TOOLKIT_ROOT_DIR` defaults to `$CUDA_HOME`, `$CUDA_PATH`, then
`/usr/local/cuda`.

## Run

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
