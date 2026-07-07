# LiDAR 版 INT8 部署可行性与测速分析

本文档记录给 **LiDAR 部署路径**（track/e2e 等）做 TensorRT INT8 显式量化
（ModelOpt Q/DQ）的可行性侦察、实际量化流程、以及 dense engine 的 INT8-vs-FP16
测速结论。动机：评估 Orin 上 FP16 可能不够快时，INT8 能带来多少加速。

结论先行（本机 SM89 实测）：

| 部件 | FP16 | INT8 | 加速 | 占 pipeline | INT8 有效? |
|---|---|---|---|---|---|
| backbone+neck 前端（纯 Conv） | 0.427 ms | 0.236 ms | **1.8×** | 小 | ✅ 有效但占比小 |
| dense engine（track head） | 7.76 ms | 7.86 ms | 无（噪声内） | **大头** | ❌ 无效 |
| sparse encoder（libspconv） | 见 §9 | — | — | 中 | ⚠️ 不走 TRT 量化 |

- **dense engine INT8 不加速**：瓶颈在**无法量化的 plugin**（18 个
  `MultiScaleDeformableAttnTRT`）+ 被排除的 attention MatMul，INT8 只吃到少量
  Gemm/Conv，加速被 Q/DQ 开销抵消。
- **前端 backbone+neck INT8 有效（1.8×）**：纯 Conv，13 Conv 全部量化。但它在整个
  pipeline 里占比小，省下的 ~0.19ms 相对 dense 的 7.76ms，对端到端 <3%。
- **总结论**：INT8 加速集中在占比小的前端，对占大头的 dense engine 无效 →
  **INT8 对整个 LiDAR pipeline 的总加速非常有限（估计 <3%）**。若 Orin 上 FP16 不够
  快，INT8 不是出路；真瓶颈是 dense engine 的 deformable attention plugin。

## 1. 可行性前提（全部就绪）

| 前提 | 状态 |
|---|---|
| TRT10.9 LiDAR plugin（量化阶段用） | ✅ `inference_app/enqueueV3/build_trt109/libuniad_plugin.so`，本会话用 TRT10.9 编译 |
| `uniad_quant` 环境 | ✅ modelopt 0.29.0 / onnx 1.16.2 / onnxruntime-gpu 1.21.0 / tensorrt 10.9.0.34 |
| TensorRT 10.9（量化）/ 10.7（engine） | ✅ 均在本机 |
| LiDAR dense ONNX | ✅ track/e2e/plan 均可导出 |

方法与相机版 `README_uniad_explicit_quantization.md` 一致：ModelOpt 在 ONNX 上插
Q/DQ 节点，TRT 按 `INT8+FP16` 混合选 tactic。量化阶段 TRT10.9，engine 构建 TRT10.7。

## 2. 校准数据（小规模，speed-first）

`tools/make_small_calib_lidar.py`：拼一个 10 样本小校准集，用于跑通量化 + 测速
（**非精度级**，精度级需两段式真实前向大校准集，见待办）。

- 真实 `lidar_bev`/`prev_bev`/`l2g`/`timestamp`：来自 10 帧 track runtime 输出的
  逐帧 dump（`frame_XXXXXX_*.bin`）。
- track-state 输入（`prev_track_intances*`）：export trace 的单帧快照，固定
  shape0=601（engine opt shape），每个样本复用，保证 concat 整齐（modelopt 要求）。
- 布局与相机版 calib npz 一致：每个输入名 N 个样本沿 axis0 concat。

产物：`UniAD/calib_data_lidar_track_small.npz`（10 样本）。

## 3. 量化过程中踩的两个 ONNX 障碍（已解决）

ModelOpt 量化前会跑一遍 ORT 推理做 MHA 排除分析，暴露了导出 ONNX 的两个问题：

1. **float64 混入 float32 图**：track ONNX 有 6 个 `Cast→DOUBLE` + 42 个
   `Constant(double)`（PyTorch export 时的 double 标量/常量，分布在各 attention 层）。
   ORT shape 推理报 `TypeInferenceError: DOUBLE vs FLOAT`。TensorRT 不支持 FP64
   （engine build 本就要降 fp32），故全部转 FLOAT 安全。工具：`tools/fix_double_casts.py`
   （覆盖 Cast / Constant / initializer / value_info 四类）。产物：`..._nodbl.onnx`。
2. **`use_prev_bev` dtype**：trace dump 是 int32，但 modelopt simplify 后的模型定义
   为 float32，校准喂 int32 会报 `Unexpected input data type`。校准脚本已改为 float32。

## 4. 量化命令与结果

`tools/quantize_track_lidar_int8.sh`（照搬相机版 recipe）：

```bash
LD_LIBRARY_PATH=$TRT109/lib:$CUDA12RT:$CUDNN9:/usr/local/cuda/lib64 \
  python -m modelopt.onnx.quantization \
    --onnx_path=UniAD/onnx/base_track_lidar_trt_epoch2_velofix_nodbl.onnx \
    --trt_plugins=inference_app/enqueueV3/build_trt109/libuniad_plugin.so \
    --calibration_eps trt cuda:0 cpu \
    --calibration_shapes=<22 inputs, shape0=601> \
    --simplify --op_types_to_exclude MatMul \
    --calibration_data_path=UniAD/calib_data_lidar_track_small.npz \
    --output_path=UniAD/onnx/base_track_lidar_trt_epoch2_velofix_int8.onnx \
    --dq_only
```

量化统计：

```text
Total number of nodes: 3612
Total number of quantized nodes: 141      # 仅 ~4% 节点量化
Quantized type counts: {Sub:7, Mul:15, Add:67, Gemm:17, MatMul:23, Conv:1, Unsqueeze:5, Shape:6}
```

注：尽管传了 `--op_types_to_exclude MatMul`，modelopt 只排除了 attention（MHA）里的
matmul，仍量化了 23 个其它 matmul。18 个 `MultiScaleDeformableAttnTRT` plugin 完全
不参与量化。

## 5. INT8 vs FP16 测速（本机 SM89，track dense engine）

engine 构建（TRT10.7，`--fp16 --int8`）：
`UniAD/engine/base_track_lidar_track_head_epoch2_int8.engine`（65 MiB，含 Q/DQ
scale + fp16 fallback 权重，比 FP16 的 46 MiB 大）。

`trtexec --loadEngine`（shape0=601，200 iters，同 plugin）：

```text
                    FP16(velofix)   INT8
GPU Compute median   7.76 ms        7.86 ms
GPU Compute mean     7.86 ms        7.99 ms
端到端 Latency median 11.26 ms       11.36 ms
```

**INT8 反而慢 ~1%（噪声内），无加速。**

## 5.5 前端 backbone+neck 的 INT8 测速（对照，INT8 有效）

前端 `base_track_lidar_backbone_neck` 是纯 Conv 网络：13 Conv + 1 ConvTranspose +
1 BN + 14 Relu，输入 `middle_bev [1,256,120,160]`（固定 shape），输出 `lidar_bev`。
无 plugin、无 attention、无 double —— INT8 最友好。

量化（`--calibration_shapes=middle_bev:1x256x120x160`，无需 plugin/无需 exclude）：

```text
Total nodes: 72   Quantized nodes: 15
Quantized type counts: {Conv:13, ConvTranspose:1, BatchNormalization:1}   # 全部量化
```

校准集 `UniAD/calib_data_frontend_small.npz`：真实 golden `middle_bev` 1 帧 + 7 份
1% 噪声副本（8 样本，speed-first）。

`trtexec --loadEngine` 测速（middle_bev 固定 shape，300 iters）：

```text
                    FP16      INT8
GPU Compute median  0.427 ms  0.236 ms    -> 1.8x 加速
```

**前端 INT8 确实加速 ~1.8×**，因为纯 Conv 全部吃到 INT8。但前端绝对耗时只有亚毫秒级，
相对 dense 的 7.76ms 占比很小，省下的 ~0.19ms 对端到端总延迟影响 <3%。

产物：
```text
UniAD/onnx/base_track_lidar_backbone_neck_epoch2_int8.onnx
UniAD/engine/base_track_lidar_backbone_neck_epoch2_int8.engine
UniAD/logs/trtexec_speed_frontend_{fp16,int8}.log
```

## 6. 结论与解读

- **dense engine（占大头，7.76ms）INT8 不加速**：瓶颈是无法量化的
  `MultiScaleDeformableAttnTRT` plugin（×18）+ 被排除的 attention matmul。INT8 只吃到
  少量 Gemm/Conv，加速被 Q/DQ 开销抵消。
- **前端 backbone+neck（占比小，0.43ms）INT8 有效（1.8×）**：纯 Conv 全量化。
- **合起来**：加速集中在占比小的前端，对占大头的 dense 无效 → **INT8 对整个 LiDAR
  pipeline 的总加速非常有限（<3%）**。若 Orin 上 FP16 不够快，INT8 不是出路。
- 结构性事实（瓶颈在 plugin）跨平台成立，故 Orin 上 dense INT8 大概率同样无收益。
  但 **SM89 ≠ Orin SM87**，INT8/FP16 相对特性不同，正式结论需 Orin 实测。

## 7. 待办 / 后续方向

- 优化 `MultiScaleDeformableAttnTRT` plugin 本身（FP16 且是 dense 真瓶颈）——这是唯一
  能实质降低 pipeline 总延迟的方向。
- 若最终要用 dense/前端 INT8：需两段式真实前向大校准集追精度（当前是 speed-first 小
  集，精度未验）。
- Orin/DriveOS 上重建 + 实测 finite/延迟。

## 8. 相关工具与产物

```text
tools/make_small_calib_lidar.py          # dense 小校准集
tools/fix_double_casts.py                 # 修 ONNX float64 -> float32
tools/quantize_track_lidar_int8.sh        # dense modelopt 量化
UniAD/calib_data_frontend_small.npz       # 前端 middle_bev 校准集
inference_app/enqueueV3/build_trt109/     # TRT10.9 LiDAR plugin
UniAD/onnx/base_track_lidar_trt_epoch2_velofix_int8.onnx           # dense INT8 ONNX
UniAD/engine/base_track_lidar_track_head_epoch2_int8.engine        # dense INT8 engine
UniAD/onnx/base_track_lidar_backbone_neck_epoch2_int8.onnx         # 前端 INT8 ONNX
UniAD/engine/base_track_lidar_backbone_neck_epoch2_int8.engine     # 前端 INT8 engine
UniAD/logs/trtexec_speed_{fp16,int8}.log                           # dense 测速
UniAD/logs/trtexec_speed_frontend_{fp16,int8}.log                  # 前端测速
```

## 9. Sparse Encoder 的量化考量（不走 TRT/ModelOpt 路径）

pipeline 最前端的 sparse voxel encoder **不是 TensorRT engine**，因此上面的 ModelOpt
Q/DQ + trtexec 量化流程**对它不适用**。事实：

- 结构：`base_track_lidar_sparse_encoder_epoch2.onnx` = 21 个 `SparseConvolution` +
  8 Add + 8 Relu + 1 ScatterDense（3D 稀疏卷积，不是稠密 Conv）。
- 运行时：由 **libspconv 1.1.10 独立 C++ runtime** 执行（`spconv/engine.hpp`），
  **不经过 TensorRT**。当前输出已是 Float16。

因此：

- **ModelOpt/trtexec 无法量化它** —— 那套只作用于 TRT engine / ONNX-to-TRT 路径。
  `SparseConvolution` 是 libspconv 的自定义 op，TRT 不认。
- 要 INT8 sparse encoder，必须走 **libspconv 自己的量化机制**（spconv 的 INT8 稀疏
  卷积支持），是另一条完全独立的工具链，本会话未涉及。
- 稀疏卷积的实际耗时取决于每帧非空 voxel 数（~12 万），是数据相关的；它在 pipeline
  里占中等比重，但量化路径独立，无法用本文档的 TRT INT8 方法评估。

**结论**：sparse encoder 的 INT8 是一个**独立的、依赖 libspconv INT8 能力**的课题，
不在 TensorRT 显式量化范围内。若要评估，需单独调研 libspconv 的 INT8 稀疏卷积支持并
准备其专用校准，属于另一项工作。

## 10. 三部件总览（INT8 适用性）

```text
部件                执行引擎           算子             INT8 路径              本会话结论
sparse encoder     libspconv(C++)     SparseConv×21    libspconv 自有(未做)    独立课题，非 TRT 量化
backbone+neck 前端  TensorRT           Conv×13          ModelOpt+trtexec       ✅ 1.8x，但占比小
dense (track head)  TensorRT           attn/plugin/MatMul ModelOpt+trtexec     ❌ 无加速(plugin瓶颈)
```

**整体判断**：能被 TensorRT INT8 有效加速的只有占比小的 backbone+neck 前端；占大头的
dense 受限于 plugin 无收益；sparse encoder 需另走 libspconv。**所以 INT8 无法显著降低
整个 LiDAR pipeline 的延迟。**
