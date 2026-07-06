#!/usr/bin/env bash
# Build a dense mapfuse engine with an EXPLICIT per-node fp32 island under
# global --fp16. The island node list (name:fp32,...) is read from a spec file
# so the giant --layerPrecisions arg lives here, not in the caller's shell.
#
# Usage: build_map_island_engine.sh <spec_file> <engine_out> [prefer|obey]
set -euo pipefail

SPEC_FILE="${1:?spec file required}"
ENGINE_OUT="${2:?engine output path required}"
CONSTRAINT="${3:-prefer}"

TRT_PATH=/home/baojiali/Downloads/TensorRT-10.7.0.23
export LD_LIBRARY_PATH="$TRT_PATH/lib:${LD_LIBRARY_PATH-}"

ONNX=UniAD/onnx/base_e2e_lidar_plan_mapfuse_trt_epoch6_fp16safe.repaired.onnx
PLUGIN=inference_app/enqueueV3/build/libuniad_plugin.so

SPEC="$(cat "$SPEC_FILE")"
NENT=$(tr ',' '\n' < "$SPEC_FILE" | grep -c ':fp32' || true)
echo "[build] spec=$SPEC_FILE entries=$NENT constraint=$CONSTRAINT"
echo "[build] engine_out=$ENGINE_OUT"

MIN=601; OPT=601; MAX=1201
TS="prev_track_intances0:Lx512,prev_track_intances1:Lx3,prev_track_intances3:L,prev_track_intances4:L,prev_track_intances5:L,prev_track_intances6:L,prev_track_intances8:L,prev_track_intances9:Lx10,prev_track_intances11:Lx4x256,prev_track_intances12:Lx4,prev_track_intances13:L"

LOG="UniAD/logs/trtexec_$(basename "${ENGINE_OUT%.engine}").log"

"$TRT_PATH/bin/trtexec" \
  --onnx="$ONNX" \
  --saveEngine="$ENGINE_OUT" \
  --staticPlugins="$PLUGIN" \
  --fp16 \
  --precisionConstraints="$CONSTRAINT" \
  --layerPrecisions="$SPEC" \
  --layerOutputTypes="$SPEC" \
  --minShapes="${TS//L/$MIN}" \
  --optShapes="${TS//L/$OPT}" \
  --maxShapes="${TS//L/$MAX}" \
  --skipInference \
  2>&1 | tee "$LOG"

echo "[build] done -> $ENGINE_OUT (log: $LOG)"
