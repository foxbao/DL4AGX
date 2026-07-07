#!/usr/bin/env bash
# ModelOpt explicit INT8 quantization for the LiDAR track dense ONNX.
# Mirrors the camera-version recipe (README_uniad_explicit_quantization.md):
#   - runs under TensorRT 10.9 (quant stage), uniad_quant env
#   - loads the TRT10.9-compiled LiDAR plugin (just built in build_trt109)
#   - excludes MatMul from INT8 (attention stays FP16), same as camera version
#   - --dq_only produces a DQ-only Q/DQ ONNX for TRT10.7 engine build
set -euo pipefail

ROOT=/home/baojiali/Downloads/public_code/DL4AGX/AV-Solutions/uniad-trt
cd "$ROOT"

PY=/home/baojiali/anaconda3/envs/uniad_quant/bin/python
TRT109=/home/baojiali/Downloads/TensorRT-10.9_x86_cu118
CUDA12RT=/home/baojiali/anaconda3/envs/detr/targets/x86_64-linux/lib
CUDNN9=/home/baojiali/anaconda3/envs/uniad_quant/lib/python3.10/site-packages/nvidia/cudnn/lib
PLUGIN=$ROOT/inference_app/enqueueV3/build_trt109/libuniad_plugin.so

ONNX=$ROOT/UniAD/onnx/base_track_lidar_trt_epoch2_velofix_nodbl.onnx
CALIB=$ROOT/UniAD/calib_data_lidar_track_small.npz
OUT=$ROOT/UniAD/onnx/base_track_lidar_trt_epoch2_velofix_int8.onnx

L=601
SHAPES=prev_track_intances0:${L}x512,prev_track_intances1:${L}x3,prev_track_intances3:${L},prev_track_intances4:${L},prev_track_intances5:${L},prev_track_intances6:${L},prev_track_intances8:${L},prev_track_intances9:${L}x10,prev_track_intances11:${L}x4x256,prev_track_intances12:${L}x4,prev_track_intances13:${L},prev_timestamp:1,prev_l2g_r_mat:1x3x3,prev_l2g_t:1x3,prev_bev:1x256x120x160,lidar_bev:1x256x120x160,shift:1x2,timestamp:1,l2g_r_mat:1x3x3,l2g_t:1x3,use_prev_bev:1,max_obj_id:1

echo "[quant] onnx=$ONNX"
echo "[quant] calib=$CALIB  plugin=$PLUGIN"

LD_LIBRARY_PATH=$TRT109/lib:$CUDA12RT:$CUDNN9:/usr/local/cuda/lib64:${LD_LIBRARY_PATH-} \
  "$PY" -m modelopt.onnx.quantization \
    --onnx_path="$ONNX" \
    --trt_plugins="$PLUGIN" \
    --calibration_eps trt cuda:0 cpu \
    --calibration_shapes="$SHAPES" \
    --simplify \
    --op_types_to_exclude MatMul \
    --calibration_data_path="$CALIB" \
    --output_path="$OUT" \
    --dq_only

echo "[quant] done -> $OUT"
ls -l "$OUT" 2>&1 | awk '{print $NF, $5}'
