# Vendored 3DSparseConvolution Runtime Subset

This directory contains the minimal runtime subset used by
`inference_app/sparse_lidar`:

- `src/onnx-parser.cpp` and `src/onnx-parser.hpp`
- ONNX proto files needed to generate parser protobuf sources
- `libspconv` headers
- prebuilt `libspconv.so` libraries for the CUDA/platform combinations already
  present in the source dependency

Large sample workspaces, logs, exported tensors, and example ONNX files from
the upstream `Lidar_AI_Solution/libraries/3DSparseConvolution` directory are
intentionally not vendored here.
