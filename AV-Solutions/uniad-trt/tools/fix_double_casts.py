#!/usr/bin/env python3
"""Rewrite Cast->DOUBLE (to=11) nodes to Cast->FLOAT (to=1) in an ONNX.

modelopt's MHA-exclusion analysis runs an ORT shape-inference pass that fails
when float64 intermediates collide with the float32 graph
(TypeInferenceError: DOUBLE vs FLOAT). The LiDAR track ONNX has 6 such casts
(Mul->Cast(double)->Add, one per attention layer). TensorRT does not support
FP64 anyway — these would be demoted to fp32/fp16 at engine build regardless —
so forcing them to float32 is safe for deployment and unblocks quantization.
"""
import argparse
import onnx
from onnx import TensorProto

ap = argparse.ArgumentParser()
ap.add_argument('inp')
ap.add_argument('out')
args = ap.parse_args()

import numpy as np
from onnx import numpy_helper

m = onnx.load(args.inp)
n_cast = n_const = n_init = n_vi = 0

for node in m.graph.node:
    # Cast -> DOUBLE  =>  Cast -> FLOAT
    if node.op_type == 'Cast':
        for attr in node.attribute:
            if attr.name == 'to' and attr.i == TensorProto.DOUBLE:
                attr.i = TensorProto.FLOAT
                n_cast += 1
    # Constant nodes holding a DOUBLE tensor  =>  FLOAT (the real source here)
    if node.op_type == 'Constant':
        for attr in node.attribute:
            if attr.name == 'value' and attr.t.data_type == TensorProto.DOUBLE:
                arr = numpy_helper.to_array(attr.t).astype(np.float32)
                new_t = numpy_helper.from_array(arr, attr.t.name)
                attr.t.CopyFrom(new_t)
                n_const += 1

# DOUBLE initializers
for init in m.graph.initializer:
    if init.data_type == TensorProto.DOUBLE:
        arr = numpy_helper.to_array(init).astype(np.float32)
        new_i = numpy_helper.from_array(arr, init.name)
        init.CopyFrom(new_i)
        n_init += 1

# DOUBLE value_info / outputs
for vi in list(m.graph.value_info) + list(m.graph.output):
    if vi.type.tensor_type.elem_type == TensorProto.DOUBLE:
        vi.type.tensor_type.elem_type = TensorProto.FLOAT
        n_vi += 1

onnx.save(m, args.out)
print(f'rewrote: Cast->FLOAT={n_cast}  Constant(double)->float={n_const}  '
      f'initializer={n_init}  value_info={n_vi}')
print(f'saved {args.out}')
