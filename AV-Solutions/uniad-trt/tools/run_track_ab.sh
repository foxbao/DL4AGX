#!/usr/bin/env bash
# A/B run of the track-lidar runtime: velofix engine vs baseline engine,
# identical inputs, two DISTINCT output dirs. Verifies files actually land.
set -euo pipefail

TRT_PATH=/home/baojiali/Downloads/TensorRT-10.7.0.23
export LD_LIBRARY_PATH="$TRT_PATH/lib:${LD_LIBRARY_PATH-}"
export CUDA_VISIBLE_DEVICES=0

BIN=inference_app/sparse_lidar/build/uniad_lidar_track
SPARSE=UniAD_train/UniAD/onnx/base_track_lidar_sparse_encoder_epoch2.onnx
BNECK=UniAD/engine/base_track_lidar_backbone_neck_epoch2.engine
PLUGIN=inference_app/enqueueV3/build/libuniad_plugin.so
DATA=UniAD_train/UniAD/dumped_inputs/base_track_lidar_deploy_data_10f
INIT=UniAD/dumped_inputs/base_track_lidar_trt_trace_epoch2
N=10

run_one() {
  local tag="$1" engine="$2" out="$3"
  echo "=================== RUN: $tag ==================="
  echo "engine=$engine"
  echo "out=$out"
  rm -rf "$out"; mkdir -p "$out"
  "$BIN" "$SPARSE" "$BNECK" "$engine" "$PLUGIN" "$DATA" "$out" "$N" \
    --metadata-json "$DATA" \
    --gt-detections "$DATA" \
    --track-init-dir "$INIT" \
    --score-threshold 0.2 \
    --bev-max-draw 200 2>&1 | tail -3
  echo "--- files landed in $out ---"
  ls -1 "$out" 2>/dev/null | wc -l
  ls -1 "$out"/frame_000009_* 2>/dev/null | head -20
}

# Both engines must be exported with the SAME toolchain/day so they differ ONLY
# by the velo_update line — otherwise fp16 tactic drift pollutes the diff. The
# _oldclean engine is the naive-formula build from parent commit e8cae5b via the
# same export path as _velofix (see tools/export_track_old_clean.sh).
run_one velofix  UniAD/engine/base_track_lidar_track_head_epoch2_velofix.engine \
  UniAD/output/track_velofix_ab_10f
run_one oldclean UniAD/engine/base_track_lidar_track_head_epoch2_oldclean.engine \
  UniAD/output/track_oldclean_ab_10f

echo "=================== DONE ==================="
