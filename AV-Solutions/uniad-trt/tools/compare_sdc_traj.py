#!/usr/bin/env python3
"""Compare sdc_traj bins across runtime output dirs.

Usage: compare_sdc_traj.py <dir_under_test> [--ref DIR ...]
Reports finiteness of the dir-under-test and, for each reference dir, the
per-frame and global max/mean abs diff of sdc_traj.
"""
import argparse
import glob
import os
import numpy as np


def load(d):
    fs = sorted(glob.glob(os.path.join(d, 'frame_*_sdc_traj.bin')))
    return fs, {os.path.basename(f): np.fromfile(f, dtype=np.float32)
                for f in fs}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('test')
    ap.add_argument('--ref', action='append', default=[])
    args = ap.parse_args()

    fs, test = load(args.test)
    allv = np.concatenate(list(test.values())) if test else np.array([])
    print(f'== under test: {args.test} ==')
    print(f'   frames={len(fs)}  finite={bool(np.isfinite(allv).all())}  '
          f'nan={int(np.isnan(allv).sum())}  inf={int(np.isinf(allv).sum())}')
    if allv.size:
        print(f'   global min/max = {allv.min():.6f} / {allv.max():.6f}')

    for ref in args.ref:
        _, r = load(ref)
        common = sorted(set(test) & set(r))
        if not common:
            print(f'== vs {ref}: NO COMMON FRAMES ==')
            continue
        diffs = np.concatenate([np.abs(test[k] - r[k]) for k in common])
        # ignore nan in diff so a nan test dir still reports structure
        finite = diffs[np.isfinite(diffs)]
        print(f'== vs {ref} ({len(common)} frames) ==')
        if finite.size:
            print(f'   max abs diff  = {finite.max():.6f} m')
            print(f'   mean abs diff = {finite.mean():.6f} m')
        print(f'   nan in diff   = {int(np.isnan(diffs).sum())}')


if __name__ == '__main__':
    main()
