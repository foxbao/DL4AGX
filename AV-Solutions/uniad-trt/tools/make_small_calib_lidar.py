#!/usr/bin/env python3
"""Build a SMALL INT8 calibration npz for the LiDAR track dense engine.

Purpose: get modelopt quantization to run so we can measure INT8-vs-FP16 speed
on this dev machine. This is a minimal, speed-first calib set — NOT a precision-
grade one (that needs the two-stage real-forward harness over many frames).

Source of real tensors:
- lidar_bev / prev_bev / l2g_* / timestamp: per-frame real dumps from the
  10-frame track runtime output dir (frame_XXXXXX_*.bin).
- track-state inputs (prev_track_intances*): the export trace dump, which is
  one real frame at fixed shape0=601 (the engine's opt shape). We reuse that
  single track-state snapshot for every calib sample — track-state input
  distribution matters less for a speed probe, and it keeps shape0 fixed at 601
  so all samples concat cleanly (modelopt requirement).

Layout matches the camera calib npz: per input name, N samples concatenated
along axis 0 (track/bev along their row dim, scalars along frame dim).
"""
import argparse
import os
import numpy as np

# name -> (dtype, per-sample shape at L=601)
L = 601
TRACK_SPECS = {
    'prev_track_intances0': (np.float32, (L, 512)),
    'prev_track_intances1': (np.float32, (L, 3)),
    'prev_track_intances3': (np.int32, (L,)),
    'prev_track_intances4': (np.int32, (L,)),
    'prev_track_intances5': (np.int32, (L,)),
    'prev_track_intances6': (np.float32, (L,)),
    'prev_track_intances8': (np.float32, (L,)),
    'prev_track_intances9': (np.float32, (L, 10)),
    'prev_track_intances11': (np.float32, (L, 4, 256)),
    'prev_track_intances12': (np.int32, (L, 4)),
    'prev_track_intances13': (np.float32, (L,)),
}
# per-sample real tensors that vary per frame (from runtime dump)
FRAME_SPECS = {
    'lidar_bev': (np.float32, (1, 256, 120, 160)),
    'prev_bev': (np.float32, (1, 256, 120, 160)),
    'prev_timestamp': (np.float32, (1,)),
    'prev_l2g_r_mat': (np.float32, (1, 3, 3)),
    'prev_l2g_t': (np.float32, (1, 3)),
    'timestamp': (np.float32, (1,)),
    'l2g_r_mat': (np.float32, (1, 3, 3)),
    'l2g_t': (np.float32, (1, 3)),
    'shift': (np.float32, (1, 2)),
    # NB: modelopt-simplified model defines use_prev_bev as float32 (not int32
    # as in the trace dump), so the calib tensor must be float32 to match.
    'use_prev_bev': (np.float32, (1,)),
    'max_obj_id': (np.int32, (1,)),
}


def load_dat(path, dtype, shape):
    a = np.fromfile(path, dtype=dtype)
    return a.reshape(shape)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--trace-dir',
                    default='UniAD/dumped_inputs/base_track_lidar_trt_trace_epoch2',
                    help='export trace dump: fixed shape0=601 track-state + one real frame')
    ap.add_argument('--runtime-dir',
                    default='UniAD_train/UniAD/output/base_track_lidar_track_epoch2_10f',
                    help='10-frame runtime output with per-frame real lidar_bev etc.')
    ap.add_argument('--num-frames', type=int, default=10)
    ap.add_argument('--out', default='UniAD/calib_data_lidar_track_small.npz')
    args = ap.parse_args()

    # 1. Fixed track-state snapshot from the trace (shape0=601), reused per sample.
    track = {}
    for name, (dt, shp) in TRACK_SPECS.items():
        p = os.path.join(args.trace_dir, name + '.dat')
        track[name] = load_dat(p, dt, shp)

    # 2. Per-frame real tensors. Prefer runtime dump; fall back to trace frame.
    def frame_tensor(fidx, name, dt, shp):
        # runtime dumps outputs as frame_XXXXXX_<name>_out.bin for track-state,
        # and frame_XXXXXX_<name>.bin for lidar_bev; l2g/timestamp use *_out.
        cand = [
            os.path.join(args.runtime_dir, f'frame_{fidx:06d}_{name}.bin'),
            os.path.join(args.runtime_dir, f'frame_{fidx:06d}_{name}_out.bin'),
        ]
        for c in cand:
            if os.path.exists(c):
                a = np.fromfile(c, dtype=dt)
                if a.size == int(np.prod(shp)):
                    return a.reshape(shp)
        # fallback: trace's single real frame
        tp = os.path.join(args.trace_dir, name + '.dat')
        if os.path.exists(tp):
            return load_dat(tp, dt, shp)
        return np.zeros(shp, dtype=dt)

    npz = {}
    n_used = 0
    for f in range(args.num_frames):
        # skip frames with no real lidar_bev
        lb = os.path.join(args.runtime_dir, f'frame_{f:06d}_lidar_bev.bin')
        if not os.path.exists(lb):
            continue
        n_used += 1
        # track-state: reuse fixed snapshot
        for name, arr in track.items():
            npz.setdefault(name, []).append(arr)
        # per-frame real tensors
        for name, (dt, shp) in FRAME_SPECS.items():
            npz.setdefault(name, []).append(frame_tensor(f, name, dt, shp))

    # 3. Concatenate along axis 0 (camera-calib layout).
    out = {}
    for name, lst in npz.items():
        out[name] = np.concatenate(lst, axis=0)

    np.savez(args.out, **out)
    print(f'wrote {args.out}: {n_used} samples')
    for k in sorted(out):
        print(f'  {k:24s} {out[k].shape} {out[k].dtype}')


if __name__ == '__main__':
    main()
