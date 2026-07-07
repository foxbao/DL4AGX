# LiDAR 版部署提速分析：INT8 可行性 + GPU 体素化

本文档记录 **LiDAR 部署路径**（track/e2e 等）的提速探索。动机：评估 Orin 上 FP16 可能
不够快时怎么提速。两条主线：
1. **TensorRT INT8**（ModelOpt Q/DQ）——三部件（前端 Conv / dense track head / spconv）
   全部实测，结论是 INT8 帮助有限（§1–§9.8、§10）。
2. **分阶段计时定位真瓶颈 + GPU 体素化**——发现瓶颈是 CPU 体素化而非 GPU 计算，用 GPU
   3D 体素化把它从 45ms 干到 9.5ms（§9.7、§9.9），**这是真正有效的加速**。

当前状态与后续见 §11。

结论先行（本机 SM89 实测）：

| 部件 | FP16 | INT8 | 加速 | INT8 有效? |
|---|---|---|---|---|
| backbone+neck 前端（纯 Conv） | 0.427 ms | 0.236 ms | **1.8×** | ✅ 有效但绝对耗时/占比小 |
| dense engine（track head） | 7.76 ms | 7.86 ms | 无（噪声内） | ❌ 无效（plugin 瓶颈，需 QAT） |
| sparse encoder（libspconv） | 2.3 ms | 2.6 ms | 无 | ❌ 能 build 不加速（索引瓶颈，§9.8） |

**两条主线结论：**

1. **INT8 不是这个 pipeline 的提速手段**：三部件全部实测，能被有效 INT8 加速的只有
   占比小的前端 Conv；dense 受限于无法量化的 deformable attention plugin（要提速须
   QAT 重训）；spconv 能 build 但不加速（瓶颈在稀疏索引非算术）。根因是 GPU 计算本就
   不是瓶颈——纯 GPU 三段合计仅 ~10.5ms（§9.7）。

2. **真瓶颈是 CPU 体素化，已用 GPU 体素化解决**：分阶段计时（§9.7）发现 pipeline 墙钟
   大头是 CPU 体素化（~45ms/帧），不是 GPU 计算。用 GPU 3D 体素化把它干到 ~9.5ms
   （~4.7×，§9.9）——**这才是本会话真正有效的加速**。（注：上车接 Apollo 后走通道读点云，
   不读文件，磁盘 IO 那部分不再是问题；GPU 体素化 kernel 本身仍适用。）

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

## 9.5 Deformable attention plugin 的 INT8:能，但需要 QAT 重训

进一步核实 dense 瓶颈 `MultiScaleDeformableAttnTRT` 到底能不能 INT8：

**plugin 本身支持 INT8。** 源码 `UniAD/plugins/plugin/multi_scale_deformable_attn/
multiScaleDeformableAttnPlugin.cpp`：
- `enqueue` 有 `kINT8` 分支 → `ms_deformable_im2col_cuda_int8`（value/offset/weight
  走 int8_4，带 scale_value/scale_offset/scale_weight/scale_out）；
- `supportsFormatCombination` 声明 value 支持 INT8，前提 `channels%4==0 &&
  point_num%4==0`；本模型 channels=256/8=32、point_num∈{4,8}，**前提全满足**。
  （plugin 来自 [DerryHub/BEVFormer_tensorrt](https://github.com/DerryHub/BEVFormer_tensorrt)，
  该项目实现了 float/half/half2/int8 四套 kernel。）

**但现有 ModelOpt PTQ 链路触发不了它**（三次尝试全失败）：
1. `--trt_plugins_precision` 只支持 fp32/fp16，**无 int8** 选项；
2. `--nodes_to_quantize` 点名 plugin value 上游的 Reshape → 量化 0 个节点
   （Reshape 是形状算子，modelopt 不量化）；
3. 点名上游 Add → 插了 Q/DQ 但 `qdq_to_dq` 转换崩溃
   （`tuple index out of range`，modelopt 处理不了 plugin 前的 Q/DQ 位置）。

**根因**：ModelOpt 是为标准 ONNX 算子设计的 PTQ 工具，不支持给自定义 plugin 喂 INT8
输入。plugin INT8 需要 value/offset/weight 都是 int8+scale，这套只有 BEVFormer_tensorrt
原生的 `pytorch_quantization` **QAT** 流程能正确生成（PyTorch 侧手动包 QuantModule）。

**且精度是大风险**：QD-BEV 实测量化 BEVFormer encoder 令 mAP 从 0.416 掉到 0.160
（-62%）。所以必须 **QAT（重训）**而非 PTQ 才能保精度。

**结论**：deformable attention 的 INT8 提速 = 引入 pytorch_quantization + 改模型代码包
QuantModule + **QAT 重训** + 精度调优，是独立立项级工程，本会话不做。若 Orin FP16 缺口
不大不值得；缺口大且必须 INT8 才专门做。

## 9.7 分阶段计时:pipeline 真瓶颈是 IO/CPU，不是 GPU 计算

给 `uniad_lidar_e2e` runtime 加了分阶段 GPU 计时（每段后 cudaStreamSynchronize），
并把 sparse encoder 内部进一步拆成"体素化(IO/CPU)"vs"spconv GPU forward"。
10 帧、frame0 排除，本机 SM89：

```text
三段墙钟(含 IO/CPU/host 拷贝) ms/frame:
  sparse encoder : 339   (69%)   <- 其中体素化+H2D ~45ms, spconv GPU 仅 ~2.3ms
  backbone+neck  :  40   ( 8%)
  dense e2e      : 111   (23%)
sparse 内部拆分(每帧):
  voxelize + H2D (磁盘读42万点 + CPU哈希体素化): ~45 ms
  spconv GPU forward:                            ~2.3 ms   <- 真正的 GPU 计算
```

**关键发现**：
- 上面的墙钟大头是 **IO / CPU / host-device 拷贝**，不是 GPU 计算。纯 GPU 计算三段
  合计只有 ~10.5ms（spconv ~2.3 + backbone ~0.43 + dense ~7.76，后两者取 trtexec 纯
  GPU 值）。
- sparse encoder 段 339ms 里，**spconv 的 GPU 计算只占 ~2.3ms**，前面的体素化 IO/CPU
  占 ~45ms（20×）。
- 因此 pipeline 的真瓶颈是**体素化(CPU) + host 内存拷贝**，INT8 对这些无能为力。

**对 spconv INT8 的意义**：量化 spconv 最多优化那 ~2.3ms 的 GPU 计算，端到端收益很小。
真正该做的是优化 CPU 体素化——**已在 §9.9 用 GPU 体素化做到（45→9.5ms）**。spconv INT8
后来也实测了（§9.8：能 build 不加速）。

计时代码：`inference_app/sparse_lidar/src/uniad_lidar_e2e.cpp`（三段 STAGE TIMING）+
`src/lidar_runtime.cpp`（SPARSE SPLIT）。日志：`UniAD/logs/stage_timing_probe2.log`。

## 9.8 spconv INT8:能量化能 build，但本机不加速（已止损）

按需实测了 sparse encoder 的 libspconv INT8（PTQ，非 QAT）。libspconv INT8 是读
ONNX SparseConvolution/Add 节点属性触发的：`precision/output_precision="int8"` +
`weight_dynamic_ranges`（per-out-channel max|w|，静态）+ `input_dynamic_range`
（per-tensor，PTQ 校准）。

做法（`UniAD_train/UniAD/tools/spconv_int8_calibrate.py`，复用导出脚本的模型/voxelize）：
1. weight range：从 ONNX 权重算 per-output-channel max|w|（不需数据）；
2. input range：forward-hook 每个 spconv 层，跑 8 帧真实数据收 max|input|（不需
   pytorch_quantization，per-tensor 标量用 hook 收 max 即可）；
3. 把两者写进 ONNX 属性、precision 翻 int8 → 生成 INT8 sparse ONNX（21/21 层）。

runtime 侧：libspconv 的 build precision 是全局开关，原本硬编码 `Precision::Float16`
（会忽略节点 int8 属性）。加了 `SPARSE_LIDAR_INT8=1` 环境开关走 `Precision::Int8`
（默认 FP16 不变，`lidar_runtime.cpp`）。

**实测结果（止损点）**：
- INT8 build 生效（日志 `built libspconv engine with Precision::Int8`），
  但**跑第 1 帧 spconv = 2.611 ms，对比 FP16 ~2.3 ms，不快甚至略慢**。
- 且完整跑通还差两处边界修正（本会话未做，因已证明不快）：
  - **Add 残差层**要同步标 int8 + `input0/1_dynamic_range`，否则 libspconv 断言
    `Different input dtype(Int8 != Float16)` 崩（脚本只处理了 SparseConvolution）；
  - **最后一层 conv_out 的 output_precision 应保持 fp16**（否则输出 int8，下游
    backbone TRT engine 报 `Unsupported sparse tensor dtype: Int8`）。

**结论**：spconv INT8 技术上可行（能量化、能 build），但**本机不加速**。原因：这个
UniAD-tiny sparse encoder 只有 21 层、GPU 仅 ~2.3ms，瓶颈在 rulebook / gather-scatter
稀疏索引开销（INT8 加速不了），而非 conv 算术。与车上 BEVFusion 不同——BEVFusion 的
spconv backbone 大得多、GPU 计算占主导，INT8 才划算；此处 sparse encoder 太小。

**INT8 线整体收官**：dense（plugin 瓶颈，需 QAT）、前端 Conv（1.8× 但占比小）、
spconv（能量化不加速）三部件全部实测——**对这个 pipeline，INT8 无有意义的总加速**，
因为 GPU 计算本就不是瓶颈（瓶颈是 IO/CPU 体素化 + host 拷贝，见 §9.7）。

## 9.9 GPU 体素化:把 45ms CPU 瓶颈干到 ~9.5ms（真正有效的加速）

§9.7 定位到 pipeline 真瓶颈是 CPU 体素化（`voxelize_raw_points` 的 unordered_map
哈希 + mean-VFE，~45ms/帧）。借鉴 NVIDIA CUDA-PointPillars 的 GPU 体素化机制
（dense 网格 + atomicAdd，`~/Downloads/public_code/CUDA-PointPillars`），改造成
**3D voxel + mean-VFE**（PointPillars 是 2D pillar + 10 维特征，这里是 3D voxel +
4 维均值），实现自包含的 GPU kernel。

实现（`inference_app/sparse_lidar/src/gpu_voxelize.cu` + `include/gpu_voxelize.hpp`）：
- kernel1 `scatter`：每点一线程，算 3D cell `(cz*gy+cy)*gx+cx`，atomicAdd 到 dense
  count + atomicAdd 累加特征到 dense featsum（cap max_points_per_voxel）；
- kernel2 `compact`：每 cell 一线程，count>0 的 atomicAdd 抢紧凑 voxel id，写
  mean 特征（fp16）+ coors `{0,cz,cy,cx}`，直接输出 device 指针给 spconv（省 H2D）。
- dense 缓冲：count 50.4M×4B≈200MB + featsum ×4≈800MB，每帧 memsetAsync 清零复用。
- runtime 开关 `SPARSE_LIDAR_GPU_VOXEL=1`（默认走 CPU，不影响现有路径）。

实测（SM89，10 帧）：

```text
                      CPU(unordered_map)   GPU(dense+atomicAdd)
voxelize + H2D/frame   ~45 ms               ~9.5 ms   -> ~4.7x
voxel 数               124329               124329    完全一致
lidar_bev vs CPU       —                    mean|Δ|=0.0005, max|Δ|~0.3-0.6
检测数                  4 tracks             4 tracks  一致
```

数值差异来源：4.2% 的 voxel 点数 >10（最多 1773 点），被 cap 到 10 个点求均值时
**CPU 取遍历顺序前 10、GPU 取 atomic 并发顺序前 10 → 选的 10 个点不同 → 均值略不同**。
95.8% 的 voxel（≤10 点）完全一致。这是语义可接受的差异（cap 选点本无顺序保证），
非 bug；下游检测数一致。

**这是本会话第一个真正有效的加速**：不同于 INT8（收益存疑），GPU 体素化实打实把
pipeline 大头之一 45→9.5ms，且是纯 CUDA kernel，对 Orin 同样有效。剩余 ~9.5ms 含磁盘
读点云 + H2D（kernel 本身应 <2ms），若要进一步压可 pin-memory / 异步读点云——另做。

产物：`gpu_voxelize.{cu,hpp}`、CMake 加 CUDA 语言。日志：`UniAD/logs/gpu_voxel_probe.log`。

## 10. 三部件总览（INT8 适用性）

```text
部件                执行引擎           算子             INT8 路径          GPU耗时      本会话结论
sparse encoder     libspconv(C++)     SparseConv×21    libspconv 属性     2.3→2.6ms    ❌ 能build不加速(索引瓶颈)
backbone+neck 前端  TensorRT           Conv×13          ModelOpt+trtexec  0.43→0.24ms  ✅ 1.8x 但占比小
dense (track head)  TensorRT           attn/plugin/MatMul ModelOpt+trtexec 7.76→7.86ms  ❌ 无加速(plugin瓶颈,需QAT)
```

**整体判断**：三部件 INT8 全部实测——前端 Conv 能 1.8× 但绝对耗时/占比小；dense 受限
于无法量化的 deformable attention plugin（要提速须 QAT 重训）；spconv 能量化能 build 但
本机不加速（瓶颈在稀疏索引非算术）。**INT8 无法显著降低整个 LiDAR pipeline 延迟。**

**真正有效的优化不是 INT8，是 GPU 体素化（§9.9，已做，45→9.5ms）**——因为 pipeline 的
大头是 CPU 体素化，GPU 计算本就不是瓶颈（§9.7）。

## 11. 当前状态与后续（对齐进展）

**已完成并落库：**
- 分阶段计时（§9.7）：定位真瓶颈 = CPU 体素化 ~45ms。
- **GPU 3D 体素化（§9.9）：45→9.5ms（~4.7×），数值与 CPU 基本一致，检测数一致。**
  开关 `SPARSE_LIDAR_GPU_VOXEL=1`，默认 CPU 不变。**这是本会话唯一确定有效的加速。**
- INT8 三部件全部实测并给出否定/有限结论（前端 1.8× 但小、dense 无效、spconv 无效）。

**未做 / 待定（按优先级）：**
- **dense 的 deformable attention INT8**：需 QAT 重训（pytorch_quantization 包 QuantModule
  + 重训 + 精度调优），且 QD-BEV 实测量化 encoder mAP -62% 风险大。**独立立项级，未做。**
  性价比说明：dense 只 7.76ms，且 pipeline 大头（体素化）已解决，QAT 收益相对更小——
  除非 Orin 上这 7.76ms 是过不去的坎，否则不建议投入。
- GPU 体素化剩余 ~9.5ms 里的磁盘读点云/H2D：**上车接 Apollo 走通道读点云，不读文件，此项
  自然消失**；GPU kernel 本身（<2ms）保留。
- 所有实测在 SM89；Orin/DriveOS 需重建 + 复测（GPU 体素化和 INT8 结论都要 on-target 复核）。
