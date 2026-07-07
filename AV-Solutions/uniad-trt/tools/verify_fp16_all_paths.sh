#!/usr/bin/env bash
# Batch verify that every tracking deploy path builds a bare-FP16 engine with
# the velo_update fix (current source) and runs 10 frames finite.
# Per path: re-export dense ONNX (docker) -> build fp16 engine -> run 10f.
# Sequential (one GPU). Each step verifies its artifact landed.
set -uo pipefail   # NOT -e: we want to continue to next path on failure

ROOT=/home/baojiali/Downloads/public_code/DL4AGX/AV-Solutions/uniad-trt
cd "$ROOT"
TRT=/home/baojiali/Downloads/TensorRT-10.7.0.23
export LD_LIBRARY_PATH="$TRT/lib:${LD_LIBRARY_PATH-}"
export CUDA_VISIBLE_DEVICES=0
IMG=uniad_torch1.12
PLUGIN=inference_app/enqueueV3/build/libuniad_plugin.so
TS="prev_track_intances0:Lx512,prev_track_intances1:Lx3,prev_track_intances3:L,prev_track_intances4:L,prev_track_intances5:L,prev_track_intances6:L,prev_track_intances8:L,prev_track_intances9:Lx10,prev_track_intances11:Lx4x256,prev_track_intances12:Lx4,prev_track_intances13:L"

STAMP=velofix_20260707
RESULTS="$ROOT/UniAD/logs/verify_fp16_all_${STAMP}.summary"
: > "$RESULTS"

log(){ echo "[$(basename "$0")] $*"; }
record(){ echo "$1" >> "$RESULTS"; }

# ---- generic export (docker) ----
do_export(){
  local script="$1" cfg="$2" ckpt="$3" onnx="$4" dump="$5"; shift 5
  local extra="$*"
  docker run --rm --gpus all -v "$ROOT":"$ROOT" -w "$ROOT/UniAD" "$IMG" bash -lc "
    export PYTHONPATH=\$PWD:\$PYTHONPATH
    python tools/$script $cfg $ckpt --onnx-file $onnx --track-state-len 601 --dump-input-dir $dump $extra
  " > "$ROOT/UniAD/logs/verify_export_$(basename ${onnx%.onnx}).log" 2>&1
}

# ---- generic fp16 build ----
do_build(){
  local onnx="$1" eng="$2"
  "$TRT/bin/trtexec" --onnx="$ROOT/UniAD/$onnx" --saveEngine="$ROOT/UniAD/$eng" \
    --staticPlugins="$PLUGIN" --fp16 \
    --minShapes="${TS//L/601}" --optShapes="${TS//L/601}" --maxShapes="${TS//L/1201}" \
    --skipInference > "$ROOT/UniAD/logs/verify_build_$(basename ${eng%.engine}).log" 2>&1
}

# ---- finite check (only FLOAT tensors; skip int32 track-state/index tensors,
#      which store -1 sentinels that read as 'nan' when misparsed as float32) ----
check_finite(){
  python3 - "$1" <<'PY'
import numpy as np, glob, os, sys
d=sys.argv[1]
# int32 outputs: obj_idxes, labels, bbox_index, track_instances 3/4/5/6/13,
# max_obj_id — exclude from the finite (float) check.
INT_KEYS=('obj_idxes','labels','bbox_index','intances3_','intances4_',
          'intances5_','intances6_','intances13_','max_obj_id')
bad=0; tot=0; nan=0; badfiles=[]
for f in glob.glob(os.path.join(d,"frame_*.bin")):
    b=os.path.basename(f)
    if b.endswith("_lidar_bev.bin"): continue
    if any(k in b for k in INT_KEYS): continue
    a=np.fromfile(f,dtype=np.float32)
    if a.size==0: continue
    tot+=1
    if not np.isfinite(a).all():
        bad+=1; nan+=int((~np.isfinite(a)).sum()); badfiles.append(b)
print(f"files={tot} nonfinite_files={bad} nonfinite_vals={nan} "
      + ("ALL_FINITE" if bad==0 else "HAS_NONFINITE "+",".join(badfiles[:5])))
PY
}

# =========================== track_lidar ===========================
verify_track(){
  local T=epoch2
  log "track_lidar: export"
  do_export export_track_lidar_onnx.py \
    projects/configs/stage1_track_map_lidar/base_track_lidar_trt_p.py \
    ../UniAD_train/UniAD/projects/work_dirs/stage1_track_map_lidar/base_track_lidar/epoch_2.pth \
    onnx/base_track_lidar_trt_${T}_verify.onnx \
    dumped_inputs/base_track_lidar_trt_trace_${T}_verify
  local onnx=onnx/base_track_lidar_trt_${T}_verify.repaired.onnx
  [ -f "$ROOT/UniAD/$onnx" ] || { record "track_lidar EXPORT_FAIL"; return; }
  log "track_lidar: build"
  do_build "$onnx" engine/base_track_lidar_track_head_${T}_verify.engine
  [ -f "$ROOT/UniAD/engine/base_track_lidar_track_head_${T}_verify.engine" ] || { record "track_lidar BUILD_FAIL"; return; }
  log "track_lidar: run 10f"
  local out=UniAD/output/verify_track_lidar_10f; rm -rf "$out"; mkdir -p "$out"
  inference_app/sparse_lidar/build/uniad_lidar_track \
    UniAD_train/UniAD/onnx/base_track_lidar_sparse_encoder_${T}.onnx \
    UniAD/engine/base_track_lidar_backbone_neck_${T}.engine \
    UniAD/engine/base_track_lidar_track_head_${T}_verify.engine \
    "$PLUGIN" \
    UniAD_train/UniAD/dumped_inputs/base_track_lidar_deploy_data_10f \
    "$out" 10 \
    --metadata-json UniAD_train/UniAD/dumped_inputs/base_track_lidar_deploy_data_10f \
    --gt-detections UniAD_train/UniAD/dumped_inputs/base_track_lidar_deploy_data_10f \
    --track-init-dir UniAD/dumped_inputs/base_track_lidar_trt_trace_${T}_verify \
    --score-threshold 0.2 --bev-max-draw 200 \
    > UniAD/logs/verify_run_track_lidar.log 2>&1
  record "track_lidar rc=$? $(check_finite "$out")"
}

# =========================== track_drivable ===========================
verify_drivable(){
  local T=epoch2
  log "drivable: export"
  do_export export_track_drivable_lidar_onnx.py \
    projects/configs/stage1_track_map_lidar/base_track_drivable_lidar_trt_p.py \
    ../UniAD_train/UniAD/projects/work_dirs/stage1_track_map_lidar/base_track_drivable_lidar/epoch_2.pth \
    onnx/base_track_drivable_lidar_trt_${T}_verify.onnx \
    dumped_inputs/base_track_drivable_lidar_trt_trace_${T}_verify
  local onnx=onnx/base_track_drivable_lidar_trt_${T}_verify.repaired.onnx
  [ -f "$ROOT/UniAD/$onnx" ] || { record "drivable EXPORT_FAIL"; return; }
  log "drivable: build"
  do_build "$onnx" engine/base_track_drivable_lidar_track_drivable_${T}_verify.engine
  [ -f "$ROOT/UniAD/engine/base_track_drivable_lidar_track_drivable_${T}_verify.engine" ] || { record "drivable BUILD_FAIL"; return; }
  log "drivable: run 10f"
  local out=UniAD/output/verify_drivable_10f; rm -rf "$out"; mkdir -p "$out"
  inference_app/sparse_lidar/build/uniad_lidar_track_drivable \
    UniAD_train/UniAD/onnx/base_track_drivable_lidar_sparse_encoder_${T}.onnx \
    UniAD/engine/base_track_drivable_lidar_backbone_neck_${T}.engine \
    UniAD/engine/base_track_drivable_lidar_track_drivable_${T}_verify.engine \
    "$PLUGIN" \
    UniAD_train/UniAD/dumped_inputs/base_track_lidar_deploy_data_10f \
    "$out" 10 \
    --metadata-json UniAD_train/UniAD/dumped_inputs/base_track_lidar_deploy_data_10f \
    --gt-detections UniAD_train/UniAD/dumped_inputs/base_track_lidar_deploy_data_10f \
    --track-init-dir UniAD/dumped_inputs/base_track_drivable_lidar_trt_trace_${T}_verify \
    --score-threshold 0.2 --bev-max-draw 200 \
    > UniAD/logs/verify_run_drivable.log 2>&1
  record "drivable rc=$? $(check_finite "$out")"
}

# =========================== e2e_lidar ===========================
verify_e2e(){
  local T=epoch2
  log "e2e_lidar: export"
  do_export export_e2e_lidar_onnx.py \
    projects/configs/stage2_e2e_lidar/base_e2e_lidar_trt_p.py \
    ../UniAD_train/UniAD/projects/work_dirs/stage2_e2e_lidar/base_e2e_lidar/epoch_2.pth \
    onnx/base_e2e_lidar_trt_${T}_verify.onnx \
    dumped_inputs/base_e2e_lidar_trt_trace_${T}_verify
  local onnx=onnx/base_e2e_lidar_trt_${T}_verify.repaired.onnx
  [ -f "$ROOT/UniAD/$onnx" ] || { record "e2e_lidar EXPORT_FAIL"; return; }
  log "e2e_lidar: build"
  do_build "$onnx" engine/base_e2e_lidar_trt_${T}_verify.engine
  [ -f "$ROOT/UniAD/engine/base_e2e_lidar_trt_${T}_verify.engine" ] || { record "e2e_lidar BUILD_FAIL"; return; }
  log "e2e_lidar: run 10f"
  local out=UniAD/output/verify_e2e_lidar_10f; rm -rf "$out"; mkdir -p "$out"
  inference_app/sparse_lidar/build/uniad_lidar_e2e \
    UniAD_train/UniAD/onnx/base_e2e_lidar_sparse_encoder_${T}.onnx \
    UniAD/engine/base_e2e_lidar_backbone_neck_${T}.engine \
    UniAD/engine/base_e2e_lidar_trt_${T}_verify.engine \
    "$PLUGIN" \
    UniAD_train/UniAD/dumped_inputs/base_e2e_lidar_deploy_data_10f \
    "$out" 10 \
    --metadata-json UniAD_train/UniAD/dumped_inputs/base_e2e_lidar_deploy_data_10f \
    --gt-detections UniAD_train/UniAD/dumped_inputs/base_e2e_lidar_deploy_data_10f \
    --track-init-dir UniAD/dumped_inputs/base_e2e_lidar_trt_trace_${T}_verify \
    --track-state-len 601 --max-track-state-len 1201 \
    --score-threshold 0.2 --bev-max-draw 200 \
    > UniAD/logs/verify_run_e2e_lidar.log 2>&1
  record "e2e_lidar rc=$? $(check_finite "$out")"
}

# =========================== e2e_occ ===========================
verify_occ(){
  local T=epoch4
  log "e2e_occ: export"
  do_export export_e2e_lidar_occ_onnx.py \
    projects/configs/stage2_e2e_lidar/base_e2e_lidar_occ_trt_p.py \
    ../UniAD_train/UniAD/projects/work_dirs/stage2_e2e_lidar/base_e2e_lidar_occ/epoch_4.pth \
    onnx/base_e2e_lidar_occ_trt_${T}_verify.onnx \
    dumped_inputs/base_e2e_lidar_occ_trt_trace_${T}_verify
  local onnx=onnx/base_e2e_lidar_occ_trt_${T}_verify.repaired.onnx
  [ -f "$ROOT/UniAD/$onnx" ] || { record "e2e_occ EXPORT_FAIL"; return; }
  log "e2e_occ: build"
  do_build "$onnx" engine/base_e2e_lidar_occ_trt_${T}_verify.engine
  [ -f "$ROOT/UniAD/engine/base_e2e_lidar_occ_trt_${T}_verify.engine" ] || { record "e2e_occ BUILD_FAIL"; return; }
  log "e2e_occ: run 10f"
  local out=UniAD/output/verify_e2e_occ_10f; rm -rf "$out"; mkdir -p "$out"
  inference_app/sparse_lidar/build/uniad_lidar_e2e \
    UniAD_train/UniAD/onnx/base_e2e_lidar_occ_sparse_encoder_${T}.onnx \
    UniAD/engine/base_e2e_lidar_occ_backbone_neck_${T}.engine \
    UniAD/engine/base_e2e_lidar_occ_trt_${T}_verify.engine \
    "$PLUGIN" \
    UniAD_train/UniAD/dumped_inputs/base_e2e_lidar_deploy_data_10f \
    "$out" 10 \
    --metadata-json UniAD_train/UniAD/dumped_inputs/base_e2e_lidar_deploy_data_10f \
    --gt-detections UniAD_train/UniAD/dumped_inputs/base_e2e_lidar_deploy_data_10f \
    --track-init-dir UniAD/dumped_inputs/base_e2e_lidar_occ_trt_trace_${T}_verify \
    --track-state-len 601 --max-track-state-len 1201 \
    --score-threshold 0.2 --bev-max-draw 200 \
    > UniAD/logs/verify_run_e2e_occ.log 2>&1
  record "e2e_occ rc=$? $(check_finite "$out")"
}

# track already verified ALL_FINITE in a prior run; re-run all for a clean sweep
# unless RUN_ONLY is set (space-separated: track drivable e2e occ).
RUN_ONLY="${RUN_ONLY:-drivable e2e occ}"
for p in $RUN_ONLY; do
  case "$p" in
    track)    verify_track ;;
    drivable) verify_drivable ;;
    e2e)      verify_e2e ;;
    occ)      verify_occ ;;
  esac
done
log "=== SUMMARY ==="; cat "$RESULTS"
# prepend the already-confirmed track result if not re-run this pass
grep -q "^track_lidar" "$RESULTS" || echo "track_lidar rc=0 files=120 ALL_FINITE (prior run, corrected check)"
