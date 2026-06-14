# UniAD-tiny ONNX Explicit Quantization 部署流程

这份文档记录 `documents/explicit_quantization.md` 里的
`ONNX Explicit Quantization with modelopt` 链路，以及本机实际验证结果。

核心结论：

- ModelOpt 量化阶段使用 TensorRT 10.9。
- TensorRT engine 构建和 C++ enqueueV3 部署阶段继续使用 TensorRT 10.7。
- 不建议把 ModelOpt 量化阶段也切到 TensorRT 10.7。本机用 10.7 试过，ModelOpt
  在构建校准 EP 时会失败；换 10.9 后同样输入可以正常生成 Q/DQ ONNX。

这条链路生成的是显式量化 ONNX，也就是带 `QuantizeLinear` / `DequantizeLinear`
节点的 Q/DQ 图。TensorRT 10.7 构建 engine 时会识别这些 Q/DQ 节点，按
`INT8(EQ)+FP16` 路径选择 tactic。它不是传统的 TensorRT calibration cache 流程。

## 当前状态

已完成并验证：

- 校准数据：`UniAD/calib_data_shape0_901.npz`，195 个 `shape0=901` 样本。
- 量化 ONNX：
  `UniAD/onnx/uniad_tiny_imgx0.25_cp_eq_dq.onnx`。
- TensorRT 10.7 engine：
  `UniAD/engine/uniad_tiny_imgx0.25_cp_eq_dq_trt10.7_sm89.engine`。
- `trtexec --iterations=100 --best` 通过，GPU Compute Time median 为
  `15.2583 ms`。
- C++ `inference_app/enqueueV3/build/uniad` 跑通 69 帧，生成 69 张可视化图。

相关日志：

```text
UniAD/logs/modelopt_trt109_full_20260612_234703.log
UniAD/logs/trtexec_eq_dq_trt10.7_sm89_20260613_002645.log
UniAD/logs/uniad_eq_dq_infer_trt10.7_sm89_69f_20260613_004131.log
```

## 0. 环境

以下命令默认从仓库根目录执行：

```bash
cd /home/baojiali/Downloads/public_code/DL4AGX/AV-Solutions/uniad-trt

export TRT109=/home/baojiali/Downloads/TensorRT-10.9_x86_cu118
export TRT107=/home/baojiali/Downloads/TensorRT-10.7.0.23
export TARGET_GPU_SM=89
```

量化阶段使用 `uniad_quant` 环境：

```bash
conda activate uniad_quant
```

这个环境需要：

```text
nvidia-modelopt==0.29.0
onnxruntime-gpu==1.21.0
tensorrt==10.9.0.34
onnx_graphsurgeon==0.6.1
onnx==1.16.2
```

注意：`onnxruntime-gpu==1.21.0` 需要 CUDA 12 runtime 和 cuDNN 9。当前机器上量化
阶段使用以下库路径：

```bash
export CUDA12RT=/home/baojiali/anaconda3/envs/detr/targets/x86_64-linux/lib
export CUDNN9=$CONDA_PREFIX/lib/python3.10/site-packages/nvidia/cudnn/lib
```

`onnx==1.21.0` 和 `modelopt==0.29.0` 不兼容，本机表现为导入
`modelopt.onnx.quantization` 时报 `onnx.reference.custom_element_types`
相关错误。固定到 `onnx==1.16.2` 后可用。

## 1. 准备 TensorRT Plugin

ModelOpt 量化阶段要加载 TensorRT 10.9 编译的 plugin：

```text
UniAD/plugins/lib/lib_uniad_plugins_trt10.9_x86_cu118.so
```

TensorRT 10.7 engine 构建和 C++ runtime 使用 enqueueV3 下的 10.7 plugin：

```text
inference_app/enqueueV3/build/libuniad_plugin.so
```

如果需要重新编译 10.7 plugin：

```bash
cmake -S inference_app/enqueueV3 -B inference_app/enqueueV3/build \
  -DTENSORRT_PATH=$TRT107 \
  -DTARGET_GPU_SM=$TARGET_GPU_SM

cmake --build inference_app/enqueueV3/build -j$(nproc)
```

## 2. 准备校准数据

校准数据来自固定 `prev_track_intances0.shape[0] == 901` 的样本。当前已生成：

```text
UniAD/calib_data_shape0_901.npz
```

如果需要重新生成：

```bash
cd UniAD
PYTHONPATH=$(pwd) python3 ./tools/prepare_calib_data.py
cd -
```

本次修正过根目录的 `tools/prepare_calib_data.py`，避免每个样本循环里重置
`npz_data`，并改为保存到当前工作目录下的
`calib_data_shape0_901.npz`。`UniAD/tools/prepare_calib_data.py` 和根目录脚本
当前逻辑保持一致。

## 3. 用 TensorRT 10.9 做 ModelOpt 显式量化

输入 ONNX 使用已经修复过的 FP32 模型：

```bash
export ONNX_PATH=$PWD/UniAD/onnx/uniad_tiny_imgx0.25_cp.repaired.onnx
export CALIB_PATH=$PWD/UniAD/calib_data_shape0_901.npz
export TRT109_PLUGIN=$PWD/UniAD/plugins/lib/lib_uniad_plugins_trt10.9_x86_cu118.so
export OUT_ONNX=$PWD/UniAD/onnx/uniad_tiny_imgx0.25_cp_eq_dq.onnx
```

ModelOpt 的校准 shape 固定为 `shape0=901`：

```bash
export SHAPES=prev_track_intances0:901x512,prev_track_intances1:901x3,prev_track_intances3:901,prev_track_intances4:901,prev_track_intances5:901,prev_track_intances6:901,prev_track_intances8:901,prev_track_intances9:901x10,prev_track_intances11:901x4x256,prev_track_intances12:901x4,prev_track_intances13:901,prev_timestamp:1,prev_l2g_r_mat:1x3x3,prev_l2g_t:1x3,prev_bev:2500x1x256,timestamp:1,l2g_r_mat:1x3x3,l2g_t:1x3,img:1x6x3x256x416,img_metas_can_bus:18,img_metas_lidar2img:1x6x4x4,command:1,use_prev_bev:1,max_obj_id:1
```

执行量化：

```bash
mkdir -p UniAD/logs

CUDA_VISIBLE_DEVICES=0 \
LD_LIBRARY_PATH=$TRT109/lib:$CUDA12RT:$CUDNN9:/usr/local/cuda/lib64:$LD_LIBRARY_PATH \
python -m modelopt.onnx.quantization \
  --onnx_path=$ONNX_PATH \
  --trt_plugins=$TRT109_PLUGIN \
  --calibration_eps trt cuda:0 cpu \
  --calibration_shapes=${SHAPES} \
  --simplify \
  --op_types_to_exclude MatMul \
  --calibration_data_path=$CALIB_PATH \
  --output_path=$OUT_ONNX \
  --dq_only \
  2>&1 | tee UniAD/logs/modelopt_trt109_full_$(date +%Y%m%d_%H%M%S).log
```

本机结果：

```text
Total number of quantized nodes: 460
Quantized type counts: {'Conv': 104, 'MaxPool': 8, 'Mul': 5, 'Add': 181, 'Gemm': 45, 'MatMul': 71, 'Unsqueeze': 16, 'Transpose': 5, 'Shape': 9, 'Gather': 1, 'Resize': 15}
Quantized onnx model is saved as .../UniAD/onnx/uniad_tiny_imgx0.25_cp_eq_dq.onnx
Number of tensors : 543
```

虽然命令里有 `--op_types_to_exclude MatMul`，ModelOpt 最终统计里仍可能出现
`MatMul`。这里按实际日志记录，不把它理解成命令失效；ModelOpt 会结合 TensorRT
EP 支持情况做 Q/DQ 插入和分区。

可选 ONNX 检查：

```bash
python - <<'PY'
import onnx
from collections import Counter

path = "UniAD/onnx/uniad_tiny_imgx0.25_cp_eq_dq.onnx"
m = onnx.load(path)
onnx.checker.check_model(m)
c = Counter(n.op_type for n in m.graph.node)
print("nodes", len(m.graph.node), "initializers", len(m.graph.initializer))
print("QuantizeLinear", c["QuantizeLinear"])
print("DequantizeLinear", c["DequantizeLinear"])
print("MultiScaleDeformableAttnTRT", c["MultiScaleDeformableAttnTRT"])
print("RotateTRT", c["RotateTRT"])
print("InverseTRT", c["InverseTRT"])
PY
```

本机检查结果：

```text
nodes 8426
initializers 6596
QuantizeLinear 377
DequantizeLinear 526
MultiScaleDeformableAttnTRT 27
RotateTRT 1
InverseTRT 1
```

## 4. 用 TensorRT 10.7 构建显式量化 Engine

engine 构建回到 TensorRT 10.7：

```bash
export TRT_PATH=$TRT107
export EQ_ONNX=$PWD/UniAD/onnx/uniad_tiny_imgx0.25_cp_eq_dq.onnx
export TRT107_PLUGIN=$PWD/inference_app/enqueueV3/build/libuniad_plugin.so
export EQ_ENGINE=$PWD/UniAD/engine/uniad_tiny_imgx0.25_cp_eq_dq_trt10.7_sm89.engine

mkdir -p UniAD/engine UniAD/logs
```

构建 profile：

```bash
export MIN=901
export OPT=901
export MAX=1150
export MINSHAPES=prev_track_intances0:${MIN}x512,prev_track_intances1:${MIN}x3,prev_track_intances3:${MIN},prev_track_intances4:${MIN},prev_track_intances5:${MIN},prev_track_intances6:${MIN},prev_track_intances8:${MIN},prev_track_intances9:${MIN}x10,prev_track_intances11:${MIN}x4x256,prev_track_intances12:${MIN}x4,prev_track_intances13:${MIN}
export OPTSHAPES=$MINSHAPES
export MAXSHAPES=prev_track_intances0:${MAX}x512,prev_track_intances1:${MAX}x3,prev_track_intances3:${MAX},prev_track_intances4:${MAX},prev_track_intances5:${MAX},prev_track_intances6:${MAX},prev_track_intances8:${MAX},prev_track_intances9:${MAX}x10,prev_track_intances11:${MAX}x4x256,prev_track_intances12:${MAX}x4,prev_track_intances13:${MAX}
```

构建并 benchmark：

```bash
CUDA_VISIBLE_DEVICES=0 \
LD_LIBRARY_PATH=$TRT_PATH/lib:/usr/local/cuda/lib64:$LD_LIBRARY_PATH \
$TRT_PATH/bin/trtexec \
  --onnx=$EQ_ONNX \
  --saveEngine=$EQ_ENGINE \
  --staticPlugins=$TRT107_PLUGIN \
  --verbose \
  --separateProfileRun \
  --profilingVerbosity=detailed \
  --tacticSources=+CUBLAS \
  --minShapes=$MINSHAPES \
  --optShapes=$OPTSHAPES \
  --maxShapes=$MAXSHAPES \
  --iterations=100 \
  --best \
  2>&1 | tee UniAD/logs/trtexec_eq_dq_trt10.7_sm89_$(date +%Y%m%d_%H%M%S).log
```

本机结果：

```text
engine size: 151M
Throughput: 42.3479 qps
Latency median: 16.5417 ms
GPU Compute Time median: 15.2583 ms
&&&& PASSED TensorRT.trtexec [TensorRT v100700]
```

`trtexec` 日志里能看到 `Int8` format 和 `i8f16_i8i32` tactic，说明 TensorRT
确实按 Q/DQ explicit quantization 路径处理了网络。

## 5. 跑 C++ enqueueV3 推理

`inference_app/enqueueV3` 的可视化会按 `info.txt` 里的相对路径读取图像：

```text
data/nuscenes/samples/...
```

因此从 `inference_app/enqueueV3` 目录运行前，需要有本地数据软链接：

```bash
ln -s ../../UniAD/data inference_app/enqueueV3/data
```

如果链接已经存在，不需要重复创建。

运行 69 帧：

```bash
mkdir -p UniAD/output/uniad_eq_dq_trt10.7_sm89_69f UniAD/logs

(
  cd inference_app/enqueueV3
  CUDA_VISIBLE_DEVICES=0 \
  LD_LIBRARY_PATH=$TRT107/lib:/usr/local/cuda/lib64:$LD_LIBRARY_PATH \
  ./build/uniad \
    ../../UniAD/engine/uniad_tiny_imgx0.25_cp_eq_dq_trt10.7_sm89.engine \
    ./build/libuniad_plugin.so \
    ../../UniAD/nuscenes_np/uniad_trt_input \
    ../../UniAD/output/uniad_eq_dq_trt10.7_sm89_69f \
    69
) 2>&1 | tee UniAD/logs/uniad_eq_dq_infer_trt10.7_sm89_69f_$(date +%Y%m%d_%H%M%S).log
```

输出：

```text
UniAD/output/uniad_eq_dq_trt10.7_sm89_69f/dumped_video_results/*.jpg
```

本机验证生成了 69 张可视化图，分辨率为 `2100x2361`。

## 6. 关于 C++ runtime 延迟

本机 `trtexec` 的固定 shape 随机输入 benchmark 很快，GPU Compute Time median 为
`15.2583 ms`。但是用真实 69 帧跑 `inference_app/enqueueV3/build/uniad` 时，
很多帧的 `enqueueV3` 计时是 6 秒级。

为了确认这不是 explicit quantization 独有问题，又用已有 FP32 engine 跑了 10 帧
对照：

```text
EQ 69f:   count 69, median 6300.931 ms, slow>1s 52
FP32 10f: count 10, median 7208.772 ms, slow>1s 6
```

因此当前判断是：真实 C++ runtime 的慢更像是动态 shape / DDS 输出路径或该输入序
列触发的行为，不是 Q/DQ ONNX 特有。文档里 NVIDIA 的性能表也是按
`trtexec --iterations=100` 的 `GPU Compute Time` 口径比较。

前 10 帧 EQ 和 FP32 可视化结果做粗略像素差，mean MAD 约 `0.288/255`，整体非常
接近。更严格的精度验证应该比较模型输出 tensor 或 planning MSE。

## 7. 10.7 量化阶段失败记录

为了确认是否可以全程使用 TensorRT 10.7，本机用一条 one-sample calibration 数据
试过 ModelOpt + TensorRT 10.7。依赖补齐后仍失败，主要错误包括：

```text
IBuilder::buildSerializedNetwork: Error Code 4: API Usage Error
Input dimensions with this name have different constant values or have contradictory IOptimizationProfile kMIN/kMAX constraints.

Could not find an implementation for RotateTRT(1)
```

同样的 one-sample calibration，用 TensorRT 10.9 和 10.9 plugin 可以生成 Q/DQ
ONNX。因此这部分保留 TensorRT 10.9 是必要的。

## 8. 常见问题

### ModelOpt 导入失败

如果报 `onnx.reference.custom_element_types` 相关错误，检查 `onnx` 版本：

```bash
python - <<'PY'
import onnx
print(onnx.__version__)
PY
```

当前验证可用版本是：

```bash
pip install --force-reinstall onnx==1.16.2
pip check
```

### ONNX Runtime 找不到 CUDA 12 或 cuDNN 9

`onnxruntime-gpu==1.21.0` 需要 CUDA 12 runtime 和 cuDNN 9。量化命令里要把这些
路径放进 `LD_LIBRARY_PATH`：

```bash
LD_LIBRARY_PATH=$TRT109/lib:$CUDA12RT:$CUDNN9:/usr/local/cuda/lib64:$LD_LIBRARY_PATH
```

`/usr/local/cuda/lib64` 仍然保留，是为了满足部分 plugin 对 CUDA 11/cuBLAS 11 的
依赖。

### `trtexec` 提示 calibrator 不会被使用

这是正常的。Q/DQ ONNX 属于 explicit quantization，TensorRT 不再使用传统
calibrator：

```text
Calibrator won't be used in explicit quantization mode.
Please insert Quantize/Dequantize layers to indicate which tensors to quantize/dequantize.
```

只要 ONNX 里已经有 `QuantizeLinear` / `DequantizeLinear` 节点即可。

### 真实 C++ runtime 和 `trtexec` 速度不一致

先用 `trtexec` 日志判断 engine 本身的固定 profile 性能，再用 C++ runtime 验证
真实输入能否完整跑通和可视化是否正常。当前 C++ runtime 的慢不应直接归因于
explicit quantization，因为 FP32 engine 对照也出现同样模式。
