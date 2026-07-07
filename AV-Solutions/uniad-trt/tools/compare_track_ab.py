#!/usr/bin/env python3
"""Rigorous A/B compare of two track runtime output dirs (velofix vs baseline).

Prints per-tensor max/mean abs diff in scientific notation (NOT %.6f, which
hides sub-microsecond fp32 differences), and treats integer outputs
(obj_idxes/labels/bbox_index) as exact-set comparisons so a changed track-id
assignment is not silently averaged away.
"""
import argparse
import glob
import os
import numpy as np


# name fragment -> numpy dtype for raw .bin reads
INT_KEYS = ('obj_idxes', 'labels', 'bbox_index', 'intances3', 'intances4',
            'intances5', 'intances6', 'intances13', 'max_obj_id')


def dtype_for(name):
    return np.int32 if any(k in name for k in INT_KEYS) else np.float32


def load(path, name):
    return np.fromfile(path, dtype=dtype_for(name))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('velofix')
    ap.add_argument('baseline')
    args = ap.parse_args()

    files = sorted(glob.glob(os.path.join(args.velofix, 'frame_*.bin')))
    worst = {}
    int_mismatch = []
    for vf in files:
        base = os.path.basename(vf)
        bf = os.path.join(args.baseline, base)
        if not os.path.exists(bf):
            continue
        name = base.split('_out')[0]
        a = load(vf, name)
        b = load(bf, name)
        if a.shape != b.shape:
            int_mismatch.append((base, f'shape {a.shape} vs {b.shape}'))
            continue
        if a.dtype == np.int32:
            if not np.array_equal(a, b):
                nd = int((a != b).sum())
                int_mismatch.append((base, f'{nd}/{a.size} ints differ'))
            continue
        if a.size == 0:
            continue
        d = np.abs(a.astype(np.float64) - b.astype(np.float64))
        denom = np.maximum(np.abs(b.astype(np.float64)), 1e-9)
        rel = (d / denom).max()
        key = ''.join(c for c in name if not c.isdigit()).replace(
            'frame__', '').strip('_')
        cur = worst.get(key, (0.0, 0.0))
        worst[key] = (max(cur[0], d.max()), max(cur[1], rel))

    print('=== float tensors: worst |Δ| across all 10 frames (abs, rel) ===')
    for k in sorted(worst):
        a, r = worst[k]
        print(f'  {k:32s} max|Δ|={a:.3e}  max_rel={r:.3e}')

    print('=== integer tensors (obj_idxes/labels/bbox_index): mismatches ===')
    if not int_mismatch:
        print('  none — all integer/track-id outputs byte-identical')
    else:
        for base, msg in int_mismatch:
            print(f'  {base}: {msg}')


if __name__ == '__main__':
    main()
