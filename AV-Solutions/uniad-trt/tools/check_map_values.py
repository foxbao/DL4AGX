#!/usr/bin/env python3
"""Check whether map coordinate initializers overflow FP16 (>65504), which
would make (points - trans2) = inf-inf = NaN in a TRT --fp16 transform.
Also identify the single sub-5000 node in the island."""
import argparse
import numpy as np
import onnx
from onnx import numpy_helper

FP16_MAX = 65504.0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('onnx')
    args = ap.parse_args()
    model = onnx.load(args.onnx)
    g = model.graph
    inits = {i.name: i for i in g.initializer}

    for key in ('model.map_lane_encoder.map_central',
                'model.map_lane_encoder.map_left',
                'model.map_lane_encoder.map_right',
                'model.map_lane_encoder.pc_range_tensor'):
        if key in inits:
            a = numpy_helper.to_array(inits[key])
            over = np.abs(a) > FP16_MAX
            print(f'{key}: shape={a.shape} dtype={a.dtype}')
            print(f'   abs min/max = {np.abs(a).min():.3f} / '
                  f'{np.abs(a).max():.3f}')
            print(f'   values overflowing FP16 (>{FP16_MAX}): '
                  f'{int(over.sum())} / {a.size}  '
                  f'({100.0*over.mean():.1f}%)')
        else:
            print(f'{key}: NOT FOUND')


if __name__ == '__main__':
    main()
