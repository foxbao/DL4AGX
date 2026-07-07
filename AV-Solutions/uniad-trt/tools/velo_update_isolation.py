#!/usr/bin/env python3
"""Decisive, TRT-free isolation test for the velo_update fp16 fix.

Uses REAL dumped ref_pts and REAL consecutive-frame ego2global (l2g) from the
track deploy metadata, so the ~2727 m global-translation regime is exactly the
one the engine sees on continuous (changed=0) frames. Compares the original
formula `A @ R + t1 - t2` against the fixed `A @ R + (t1 - t2)_fp32` in fp16,
against an fp64 ground truth. This isolates the formula change from any
TensorRT tactic differences between the 6/12 and today ONNX exports.
"""
import json
import os
import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TRACE = os.path.join(
    ROOT, 'UniAD/dumped_inputs/base_track_lidar_trt_trace_epoch2')
META = os.path.join(
    ROOT, 'UniAD_train/UniAD/dumped_inputs/base_track_lidar_deploy_data_10f')
PC = np.array([-64.0, -48.0, -2.0, 64.0, 48.0, 6.0], dtype=np.float64)


def load_ref_pts():
    p = os.path.join(TRACE, 'prev_track_intances1.dat')
    a = np.fromfile(p, dtype=np.float32).reshape(-1, 3)
    return a


def ego2global(frame, slot):
    """Metadata JSON is [ {'0':..,'1':..,'2':..,'3':..,'4':cur} ] — a 5-frame
    temporal queue; slot '4' is the current frame, '3' the previous one."""
    with open(os.path.join(META, f'img_metas_{frame:06d}.json')) as f:
        m = json.load(f)
    e = np.array(m[0][str(slot)]['ego2global'], dtype=np.float64).reshape(4, 4)
    R = e[:3, :3].copy()
    t = e[:3, 3].copy()
    return R, t


def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))


def denorm(ref):
    # ref: (...,3) in sigmoid space -> metric ego coords
    r = sigmoid(ref).copy()
    for i in range(3):
        r[..., i] = r[..., i] * (PC[3 + i] - PC[i]) + PC[i]
    return r


def core(ref_metric, R1, t1, R2, t2, dtype):
    """One velo_update global round-trip in the given dtype.
    R1/R2 are l2g rotations (ego->global); g2l = inv(R2).
    NOTE: model uses `ref @ R1` (row-vector convention)."""
    A = ref_metric.astype(dtype)
    R1 = R1.astype(dtype); R2 = R2.astype(dtype)
    t1 = t1.astype(dtype); t2 = t2.astype(dtype)
    return A @ R1, R1, R2, t1, t2  # helper returns pieces; see runners


def run_old(ref_metric, R1, t1, R2, t2, dt, g2l):
    A = ref_metric.astype(dt)
    glob = A @ R1.astype(dt) + t1.astype(dt) - t2.astype(dt)   # cancel in dt
    loc = glob @ g2l.astype(dt)
    return loc


def run_new(ref_metric, R1, t1, R2, t2, dt, g2l):
    A = ref_metric.astype(dt)
    delta = (t1.astype(np.float32) - t2.astype(np.float32)).astype(dt)  # fp32 first
    glob = A @ R1.astype(dt) + delta
    loc = glob @ g2l.astype(dt)
    return loc


def main():
    ref = load_ref_pts().astype(np.float64)
    ref_metric = denorm(ref)                       # (601,3) metric ego
    # continuous-frame regime: t1=prev(frame k-1), t2=curr(frame k)
    # continuous-frame regime within one metadata file: slot 3=prev, 4=curr
    R1, t1 = ego2global(5, 3)                       # prev (~2717m abs)
    R2, t2 = ego2global(5, 4)                       # curr (~0.88m apart)
    g2l = np.linalg.inv(R2)

    print(f'abs |t2| ~ {np.linalg.norm(t2):.1f} m   |t1-t2| = '
          f'{np.linalg.norm(t1 - t2):.4f} m')

    truth = run_old(ref_metric, R1, t1, R2, t2, np.float64, g2l)  # fp64 old==new
    truth_n = run_new(ref_metric, R1, t1, R2, t2, np.float64, g2l)
    print(f'[fp64] new vs old identity: max|Δ|={np.abs(truth-truth_n).max():.3e}')

    old16 = run_old(ref_metric, R1, t1, R2, t2, np.float16, g2l)
    new16 = run_new(ref_metric, R1, t1, R2, t2, np.float16, g2l)
    eo = np.abs(old16 - truth)
    en = np.abs(new16 - truth)
    print(f'[fp16] OLD err vs truth: max={eo.max():.4f}  mean={eo.mean():.4f} m')
    print(f'[fp16] NEW err vs truth: max={en.max():.4f}  mean={en.mean():.4f} m')
    if en.mean() > 0:
        print(f'       improvement: {eo.mean()/en.mean():.1f}x mean')


if __name__ == '__main__':
    main()
