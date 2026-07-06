# Mapfuse TensorRT FP16 NaN 问题分析

本文档记录 `base_e2e_lidar_plan_mapfuse.py` 部署时，TensorRT dense planning
engine 在裸 `--fp16` 下输出 `sdc_traj = NaN` 的问题、排查过程、当前可用修复方案
和后续可继续优化的方向。

## 1. 问题摘要

`base_e2e_lidar_plan_mapfuse.py` 相比普通 `base_e2e_lidar_plan.py` 多了静态 HD-map
lane 分支：

```text
MapLaneEncoderTRT
  -> lane_query / lane_query_pos / lane_valid / lane_centroids
  -> PlanningHeadSingleModeTRT._apply_map_lane_attention_trt()
  -> ego planning sdc_traj
```

裸 `--fp16` TensorRT dense mapfuse engine 可以 build 成功，但运行 10 帧 runtime 后，
`frame_XXXXXX_sdc_traj.bin` 全部为 `NaN`。同一模型在 PyTorch dense forward 下正常；
TensorRT full-FP32 dense engine 也正常。

因此问题不是 C++ runtime 写文件错误，也不是 ONNX export 完全错误，而是
**TensorRT FP16 下 mapfuse planning 分支存在数值不稳定 / 溢出传播**。

## 2. 相关产物

### Config / checkpoint

```text
config:
  UniAD_train/UniAD/projects/configs/stage2_e2e_lidar/base_e2e_lidar_plan_mapfuse.py
TRT config:
  UniAD/projects/configs/stage2_e2e_lidar/base_e2e_lidar_plan_mapfuse_trt_p.py
checkpoint:
  UniAD_train/UniAD/projects/work_dirs/stage2_e2e_lidar/base_e2e_lidar_plan_mapfuse/epoch_6.pth
```

### ONNX / engine

```text
front sparse ONNX:
  UniAD_train/UniAD/onnx/base_e2e_lidar_plan_mapfuse_sparse_encoder_epoch6.onnx
front backbone+neck engine:
  UniAD/engine/base_e2e_lidar_plan_mapfuse_backbone_neck_epoch6.engine

dense ONNX after FP16-safety code patch:
  UniAD/onnx/base_e2e_lidar_plan_mapfuse_trt_epoch6_fp16safe.repaired.onnx
recommended dense engine (localized 69-node map island, see section 9):
  UniAD/engine/base_e2e_lidar_plan_mapfuse_trt_epoch6_mapisland69.engine
fp32 spec for that engine (node names, bound to the ONNX above):
  UniAD/onnx/base_e2e_lidar_plan_mapfuse_trt_epoch6_fp16safe.mapisland69.fp32spec.txt
earlier wildcard engine (1101-node hammer, superseded):
  UniAD/engine/base_e2e_lidar_plan_mapfuse_trt_epoch6_fp16safe_attnfp32.engine
fallback dense engine:
  UniAD/engine/base_e2e_lidar_plan_mapfuse_trt_epoch6_fp32.engine
```

### Runtime / logs

```text
recommended runtime output:
  UniAD_train/UniAD/output/base_e2e_lidar_plan_mapfuse_epoch6_10f_attnfp32_20260706/
recommended runtime log:
  UniAD/logs/uniad_lidar_e2e_plan_mapfuse_epoch6_10f_attnfp32_20260706.log
recommended trtexec log:
  UniAD/logs/trtexec_base_e2e_lidar_plan_mapfuse_trt_epoch6_fp16safe_attnfp32.log
```

## 3. 现象复现

裸 `--fp16` dense engine：

```text
UniAD/engine/base_e2e_lidar_plan_mapfuse_trt_epoch6.engine
```

现象：

```text
build:       success
runtime:     success
detections:  finite
traj:        finite
sdc_traj:    all NaN across 10 frames
```

关键对照：

```text
PyTorch dense forward: normal, sdc_traj finite
TensorRT full FP32:    normal, sdc_traj finite
TensorRT bare FP16:    abnormal, sdc_traj NaN
```

这说明 NaN 更可能来自 TensorRT FP16 kernel / graph fusion / precision selection，而不是
checkpoint、本地 runtime 输出解析或 planning 输出 layout。

## 4. 排查过程

### 4.1 初始怀疑：FP16 mask sentinel 溢出

部署侧 map lane 分支里有 invalid-lane masking：

```python
dist = dist.masked_fill(~valid, 1.0e8)
lane_dist = lane_dist.masked_fill(~lane_active, 1.0e8)
```

`1.0e8` 超过 FP16 最大有限值，转换到 FP16 会变成 `inf`。这确实是一个危险点：
`inf` 进入 `TopK`、attention 或后续差分/归一化后可能传播为 `NaN`。

已做代码修复：

```text
UniAD/projects/mmdet3d_plugin/uniad/dense_heads/motion_head_plugin/map_lane_encoder.py
UniAD/projects/mmdet3d_plugin/uniad/dense_heads/planning_head.py
```

修复方式：

```python
torch.where(valid, dist, torch.full_like(dist, 1.0e4))
torch.where(lane_active, lane_dist, torch.full_like(lane_dist, 1.0e4))
```

结论：

```text
仅修复 1.0e8 -> 1.0e4 后，裸 FP16 dense engine 仍然 sdc_traj 全 NaN。
```

因此 `1.0e8` 是真实风险点，值得保留修复，但不是唯一根因。

### 4.2 第二轮：只强制 LayerNorm 相关算子 FP32

TensorRT build 日志提示：

```text
Detected layernorm nodes in FP16.
Running layernorm after self-attention with FP16 Reduce or Pow may cause overflow.
```

因此尝试只强制：

```text
ReduceMean*:fp32
Pow*:fp32
Sqrt*:fp32
```

结论：

```text
engine build success
runtime success
sdc_traj 仍然全 NaN
```

这说明 NaN 不只来自 LayerNorm 的 reduce/pow/sqrt，attention / MLP 的矩阵乘、
softmax 或后续除法同样参与了不稳定传播。

### 4.3 第三轮：attention / MLP / normalization 计算岛 FP32

最终可用方案是在整体 `--fp16` 下，对以下算子族施加 FP32 约束：

```text
MatMul*:fp32
Gemm*:fp32
Softmax*:fp32
ReduceMean*:fp32
Pow*:fp32
Sqrt*:fp32
Div*:fp32
```

结果：

```text
engine build success
10-frame runtime success
sdc_traj all finite
```

这说明当前问题主要位于 mapfuse planning 分支里的 attention / MLP / normalization
混合计算岛，而不是单个 mask 常量或单个 LayerNorm 节点。

## 5. 当前推荐修复方案

### 5.1 保留代码修复

保留 `1.0e8 -> 1.0e4` 的修复，避免 invalid lane sentinel 在 FP16 下直接变成
`inf`。虽然这不是充分条件，但它是必要的数值卫生修复。

### 5.2 使用 FP16 + FP32 attention island 构建 dense engine

推荐 dense engine 构建命令：

```bash
TRT_PATH=/home/baojiali/Downloads/TensorRT-10.7.0.23
export LD_LIBRARY_PATH="$TRT_PATH/lib:${LD_LIBRARY_PATH-}"
MIN=601
OPT=601
MAX=1201
TRACK_SHAPES="prev_track_intances0:Lx512,prev_track_intances1:Lx3,prev_track_intances3:L,prev_track_intances4:L,prev_track_intances5:L,prev_track_intances6:L,prev_track_intances8:L,prev_track_intances9:Lx10,prev_track_intances11:Lx4x256,prev_track_intances12:Lx4,prev_track_intances13:L"

"$TRT_PATH/bin/trtexec" \
  --onnx=UniAD/onnx/base_e2e_lidar_plan_mapfuse_trt_epoch6_fp16safe.repaired.onnx \
  --saveEngine=UniAD/engine/base_e2e_lidar_plan_mapfuse_trt_epoch6_fp16safe_attnfp32.engine \
  --staticPlugins=inference_app/enqueueV3/build/libuniad_plugin.so \
  --fp16 \
  --precisionConstraints=prefer \
  --layerPrecisions=MatMul*:fp32,Gemm*:fp32,Softmax*:fp32,ReduceMean*:fp32,Pow*:fp32,Sqrt*:fp32,Div*:fp32 \
  --layerOutputTypes=MatMul*:fp32,Gemm*:fp32,Softmax*:fp32,ReduceMean*:fp32,Pow*:fp32,Sqrt*:fp32,Div*:fp32 \
  --minShapes=${TRACK_SHAPES//L/${MIN}} \
  --optShapes=${TRACK_SHAPES//L/${OPT}} \
  --maxShapes=${TRACK_SHAPES//L/${MAX}} \
  --skipInference \
  2>&1 | tee UniAD/logs/trtexec_base_e2e_lidar_plan_mapfuse_trt_epoch6_fp16safe_attnfp32.log
```

### 5.3 Runtime 命令

```bash
TRT_PATH=/home/baojiali/Downloads/TensorRT-10.7.0.23
export LD_LIBRARY_PATH="$TRT_PATH/lib:${LD_LIBRARY_PATH-}"
OUT=UniAD_train/UniAD/output/base_e2e_lidar_plan_mapfuse_epoch6_10f_attnfp32_20260706
LOG=UniAD/logs/uniad_lidar_e2e_plan_mapfuse_epoch6_10f_attnfp32_20260706.log

rm -rf "$OUT"
inference_app/sparse_lidar/build/uniad_lidar_e2e \
  UniAD_train/UniAD/onnx/base_e2e_lidar_plan_mapfuse_sparse_encoder_epoch6.onnx \
  UniAD/engine/base_e2e_lidar_plan_mapfuse_backbone_neck_epoch6.engine \
  UniAD/engine/base_e2e_lidar_plan_mapfuse_trt_epoch6_fp16safe_attnfp32.engine \
  inference_app/enqueueV3/build/libuniad_plugin.so \
  UniAD_train/UniAD/dumped_inputs/base_e2e_lidar_plan_mapfuse_deploy_data_10f \
  "$OUT" 10 \
  --metadata-json UniAD_train/UniAD/dumped_inputs/base_e2e_lidar_plan_mapfuse_deploy_data_10f \
  --gt-detections UniAD_train/UniAD/dumped_inputs/base_e2e_lidar_plan_mapfuse_deploy_data_10f \
  --track-init-dir UniAD/dumped_inputs/base_e2e_lidar_plan_mapfuse_trt_trace_epoch6_fp16safe \
  --track-state-len 601 \
  --max-track-state-len 1201 \
  --command 2 \
  --score-threshold 0.2 \
  --bev-max-draw 200 \
  2>&1 | tee "$LOG"
```

## 6. 验证结果

### 6.1 attnfp32 输出

```text
output:
  UniAD_train/UniAD/output/base_e2e_lidar_plan_mapfuse_epoch6_10f_attnfp32_20260706/
sdc_traj files:
  10
all finite:
  true
global min/max:
  -0.150634765625 / 6.6015625
```

抽查：

```text
frame_000000 first=[0.7432, -0.0188], last=[4.3359, -0.1506]
frame_000001 first=[0.8462,  0.0027], last=[5.0781, -0.0136]
frame_000005 first=[0.9360, -0.0082], last=[5.5703, -0.0786]
frame_000009 first=[1.0918,  0.0257], last=[6.6016,  0.1272]
```

### 6.2 和 full FP32 对比

对比对象：

```text
UniAD_train/UniAD/output/base_e2e_lidar_plan_mapfuse_epoch6_10f_full_20260706/
```

对比结果：

```text
vs full-FP32 max abs diff:  0.0912 m
vs full-FP32 mean abs diff: 0.0165 m
```

该差异量级在 10 帧 smoke test 上可接受；后续如果要作为正式性能/精度结论，需要扩大到
更多 scene 和完整指标。

### 6.3 可视化输出

统一可视化输出位于：

```text
UniAD_train/UniAD/output/base_e2e_lidar_plan_mapfuse_epoch6_10f_attnfp32_20260706/
```

保留 6 个固定画布 WebM：

```text
base_e2e_lidar_plan_mapfuse_10f_epoch6_attnfp32_20260706_pred_fixed.webm
base_e2e_lidar_plan_mapfuse_10f_epoch6_attnfp32_20260706_compare_fixed.webm
base_e2e_lidar_plan_mapfuse_10f_epoch6_attnfp32_20260706_motion_fixed.webm
base_e2e_lidar_plan_mapfuse_10f_epoch6_attnfp32_20260706_motion_compare_fixed.webm
base_e2e_lidar_plan_mapfuse_10f_epoch6_attnfp32_20260706_planning_fixed.webm
base_e2e_lidar_plan_mapfuse_10f_epoch6_attnfp32_20260706_all_fixed.webm
```

规格：

```text
1200x900, 2 fps
```

## 7. 当前结论（已被第 9 节的定位实验修正）

> 本节保留最初的描述作为记录，但它偏笼统，且已被第 9 节推翻。当时的说法是：

```text
mapfuse dense planning 分支在 TensorRT 裸 FP16 下存在 attention/MLP/normalization
计算岛数值不稳定。单纯修复 invalid lane sentinel 或仅强制 LayerNorm reduce/pow/sqrt
为 FP32 不足以消除 NaN；必须把 MatMul/Gemm/Softmax/ReduceMean/Pow/Sqrt/Div 这一组
attention island 约束为 FP32，才能在保留整体 FP16 engine 的同时得到有限 sdc_traj。
```

**修正后的结论（见第 9 节实验）**：NaN 的根因是 **map-local** 的，不是"共享 attention
island 不稳定"。全图通配 `MatMul*/.../Div*`（实际强制 1101 层 FP32）之所以有效，
只是因为它顺带把真正的 map 分支节点也罩进去了。真正需要 FP32 的只有 map 分支内的
**69 个浮点节点**（map 几何变换 + map cross-attention + 其 LayerNorm + gate）。共享 BEV
attention (`planning_head.attn_module`) 全程保留 FP16，`sdc_traj` 仍然全 finite。

## 8. 后续建议

### 8.1 缩小 FP32 island 范围（已完成，见第 9 节）

原计划把全图通配缩小到 map 分支具体节点。**这一步已经做完**：用 ONNX 拓扑污点分析
（`tools/map_branch_recon.py`）从 map 权重初始化器前向 taint、以共享 `attn_module`/
`reg_branch` 为吸收壁，切出 map 分支，再取 `{MatMul,Gemm,Softmax,ReduceMean,Pow,Sqrt,
Div}` 交集，得到 **69 个节点**。逐节点名 FP32 约束后 `sdc_traj` 全 finite，且比 1101 层
大锤更接近 full-fp32。详见第 9 节。

### 8.2 尝试 ONNX opset 17 / INormalizationLayer

TensorRT 日志提示 LayerNorm FP16 reduce/pow 可能溢出。后续可以尝试更高 ONNX opset
或导出成 TensorRT 更容易识别的 normalization 层。不过本次实验显示仅修 LayerNorm
不够，因此即使改善 LayerNorm，也仍需关注 attention softmax / matmul。

### 8.3 增加更多验证数据

当前验证是 10 帧 smoke test。正式采用前建议扩大验证：

```text
1. 多 scene runtime
2. full FP32 vs attnfp32 trajectory diff 分布
3. planning L2 / collision 等指标（如果数据中有 planning GT）
4. runtime latency / memory 对比
```

### 8.4 保留 full FP32 fallback

当前保留：

```text
UniAD/engine/base_e2e_lidar_plan_mapfuse_trt_epoch6_fp32.engine
```

如果后续发现 mapisland69 在更大数据集上仍有个别 NaN 或指标异常，可以回退到 full FP32
dense engine 继续部署验证。

## 9. 定位实验：map 分支 69 节点 FP32 足以消除 NaN（推翻第 7 节初始结论）

### 9.1 动机与反证

第 7 节的初始结论把 NaN 归因于"共享 attention/MLP/normalization 计算岛不稳定"。但有
一条反证：**base_e2e_lidar_plan（不带 map 分支）用完全相同的裸 `--fp16` recipe 构建，
跑同一套共享 attention/MLP/norm 主干，`sdc_traj` 全 finite**。

```text
base-plan 裸 FP16 sdc_traj（UniAD_train/.../base_e2e_lidar_plan_epoch1_10f_full_20260706）
  frames=10  all_finite=True  nan_frames=0  min/max=-0.073/4.711
```

如果真是共享 attention island 的问题，base-plan 也应 NaN。它没有。唯一 confound 是
epoch 不同（base=epoch1, mapfuse=epoch6），需要同 ONNX 实验去除。

### 9.2 定位方法

`tools/map_branch_recon.py`：从消费 map 分支初始化器（`map_lane_encoder.*`、
`planning_head.map_attn_module.*`、`map_delta_proj`、`map_gate`）的节点前向 taint，以
共享 `planning_head.attn_module` / `reg_branch` 为吸收壁停止传播，得到 map 分支 = 412 个
浮点+整型节点；再取 hammer 族 `{MatMul,Gemm,Softmax,ReduceMean,Pow,Sqrt,Div}` 交集 =
**69 个节点**。ONNX 初始化器保留了完整 PyTorch 参数名，节点名（如 `Softmax_19179`）
逐字烧进 ONNX，可直接用于 `--layerPrecisions`。

op 分布印证了这就是 map 数据流：2×TopK（lane 选 top-64 + local-k=16）、2×Where（两处
sentinel）、2×Softmax + MatMul（map cross-attention）、1×ScatterElements（keep mask）、
1×CumSum（s_norm）、2×Sin/2×Cos（pos2posemb2d）。

排除的假设：map 全局坐标量级 ~2000–3340，**不溢出** FP16（65504），所以不是坐标 inf。
但 `_transform_to_ego` 的 `points - trans2`（两个 ~3000 相减）是灾难性抵消，正好落在 69
节点内，FP32 化即可治。

### 9.3 构建与运行

```bash
bash tools/build_map_island_engine.sh \
  UniAD/onnx/base_e2e_lidar_plan_mapfuse_trt_epoch6_fp16safe.mapisland69.fp32spec.txt \
  UniAD/engine/base_e2e_lidar_plan_mapfuse_trt_epoch6_mapisland69.engine prefer
```

trtexec 全部 69 层 `Set layer ... to precision fp32` 接受，build PASSED。runtime 命令与
第 5.3 节一致，仅把 dense engine 换成 `..._mapisland69.engine`，输出目录：

```text
UniAD_train/UniAD/output/base_e2e_lidar_plan_mapfuse_epoch6_10f_mapisland69_20260706/
```

### 9.4 结果对比

```text
engine            fp32层数  大小(MiB)  finite  vs full-fp32 max/mean(m)
full-fp32         全部       164.5      ✓       — (基准)
attnfp32(大锤)    1101        93.2      ✓       0.0912 / 0.0165
mapisland69       69          90.2      ✓       0.0676 / 0.0141
```

mapisland69 vs attnfp32：max 0.0508 / mean 0.0055 m。

三点结论：
1. **根因是 map-local**：只 FP32 掉 map 分支 69 个浮点节点，NaN 消失；共享 BEV attention
   全程 FP16。第 7 节"共享 attention island 不稳定"被推翻。
2. **epoch confound 排除**：同一 epoch6 ONNX，只改 FP32 范围，69 节点即足。base-plan
   之前 finite 是因为它根本没有 map 分支，不是 epoch1 权重更温和。
3. **69 是更优解**：它 vs full-fp32 的 max/mean diff 都比大锤更小——大锤把 motion/occ
   深处上千个可用 FP16 的算子也摁成 FP32，反而引入了偏离 full-fp32 的 tactic 路径。

### 9.5 目标平台（Orin SM87 / DriveOS）注意事项

本实验在开发机 SM89 + TensorRT-10.7.0.23 (x86) 上完成，最终部署在 Orin SM87 + DriveOS
TensorRT。因此：

- **可迁移物是 recipe 不是二进制**：`.engine` 跨 SM 和跨 TRT build 不可移植，必须在 Orin
  重建。真正迁移的是 ONNX + fp32 节点名 spec + trtexec flags。
- **节点名 spec 绑定此 ONNX**：`Softmax_19179` 等名字由本次 torch→onnx 导出烧定；只要
  Orin 用同一 ONNX，逐字可用。一旦重新导出 ONNX，后缀漂移、spec 作废，需用
  `tools/map_branch_recon.py` 对新 ONNX 重新生成。
- **FP16 kernel/tactic 跨平台不同**：SM87/DriveOS 的可用 tactic 与 SM89-x86 不同一套，
  NaN 源可能位移。**finite 判定必须在 Orin 上用实际 `sdc_traj.bin` 复核**，不能用本机结论
  替代。建议 Orin 上从较保守档起步，留余量。
- **约束模式**：`--precisionConstraints=prefer` 是建议非强制；DriveOS-TRT 若某节点无 FP32
  kernel 会忽略约束。map 岛只有 69 节点，可考虑对这一小簇改用 `obey` 强制，build 失败即
  暴露问题，比 prefer 更确定。
- full-fp32 fallback 同样需在 Orin 重建。

