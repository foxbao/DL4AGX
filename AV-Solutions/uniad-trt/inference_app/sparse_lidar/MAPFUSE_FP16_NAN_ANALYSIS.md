# Mapfuse TensorRT FP16 NaN 问题分析

本文档记录 `base_e2e_lidar_plan_mapfuse.py` 部署时，TensorRT dense planning
engine 在旧 ONNX 裸 `--fp16` 下输出 `sdc_traj = NaN` 的问题、排查过程、根因定位
和当前推荐的 origin-shift 裸 FP16 部署方案。

## 1. 问题摘要

`base_e2e_lidar_plan_mapfuse.py` 相比普通 `base_e2e_lidar_plan.py` 多了静态 HD-map
lane 分支：

```text
MapLaneEncoderTRT
  -> lane_query / lane_query_pos / lane_valid / lane_centroids
  -> PlanningHeadSingleModeTRT._apply_map_lane_attention_trt()
  -> ego planning sdc_traj
```

未做 origin-shift 修复的裸 `--fp16` TensorRT dense mapfuse engine 可以 build 成功，但运行 10 帧 runtime 后，
`frame_XXXXXX_sdc_traj.bin` 全部为 `NaN`。同一模型在 PyTorch dense forward 下正常；
TensorRT full-FP32 dense engine 也正常。

因此问题不是 C++ runtime 写文件错误，也不是 ONNX export 完全错误，而是
**TensorRT FP16 下 map 分支全局坐标变换存在大数相减的灾难性抵消**。修复方式不是把
map 分支永久改成 FP32，而是在 `MapLaneEncoderTRT` 中把静态 HD-map 坐标先做
origin-shift，使进入 FP16 图内的坐标量级从 ~3000 m 降到百米量级。修复后，dense
mapfuse engine 可用裸 `--fp16`、零 FP32 layer 约束运行，`sdc_traj` 全 finite。

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

recommended dense ONNX after origin-shift:
  UniAD/onnx/base_e2e_lidar_plan_mapfuse_trt_epoch6_originshift.repaired.onnx
recommended dense engine (bare FP16, zero FP32 constraints):
  UniAD/engine/base_e2e_lidar_plan_mapfuse_trt_epoch6_originshift_barefp16.engine
fallback dense engine:
  UniAD/engine/base_e2e_lidar_plan_mapfuse_trt_epoch6_fp32.engine
```

### Runtime / logs

```text
recommended runtime output:
  UniAD_train/UniAD/output/base_e2e_lidar_plan_mapfuse_epoch6_10f_originshift_barefp16_20260706/
recommended runtime log:
  UniAD/logs/uniad_lidar_e2e_plan_mapfuse_epoch6_10f_originshift_barefp16_20260706.log
recommended trtexec log:
  UniAD/logs/trtexec_base_e2e_lidar_plan_mapfuse_trt_epoch6_originshift_barefp16.log
```

## 3. 现象复现

旧 ONNX 裸 `--fp16` dense engine：

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

这说明 NaN 来自 TensorRT FP16 数值路径，而不是 checkpoint、本地 runtime 输出解析或
planning 输出 layout。后续第 9-10 节进一步证明，真正根因是 map 分支内全局坐标
`points - ego_translation` 的 FP16 大数抵消。

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

### 4.3 第三轮：attention / MLP / normalization 计算岛 FP32（历史中间方案）

在 root-cause 修复前，当时可用的中间方案是在整体 `--fp16` 下，对以下算子族施加 FP32
约束：

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

这说明全图通配 FP32 约束确实覆盖到了导致 NaN 的路径，但不能据此断定 attention 本身是
根因。第 9-10 节后续实验进一步把问题定位到 map 分支全局坐标变换的 FP16 大数抵消；
origin-shift 后不再需要这些 FP32 layer 约束。

## 5. 当前推荐修复方案：origin-shift 裸 FP16

### 5.1 保留代码修复

保留 `1.0e8 -> 1.0e4` 的修复，避免 invalid lane sentinel 在 FP16 下直接变成
`inf`。虽然这不是充分条件，但它是必要的数值卫生修复。

### 5.2 使用 origin-shift 后的裸 FP16 dense engine

推荐 dense engine 构建命令：

```bash
TRT_PATH=/home/baojiali/Downloads/TensorRT-10.7.0.23
export LD_LIBRARY_PATH="$TRT_PATH/lib:${LD_LIBRARY_PATH-}"
MIN=601
OPT=601
MAX=1201
TRACK_SHAPES="prev_track_intances0:Lx512,prev_track_intances1:Lx3,prev_track_intances3:L,prev_track_intances4:L,prev_track_intances5:L,prev_track_intances6:L,prev_track_intances8:L,prev_track_intances9:Lx10,prev_track_intances11:Lx4x256,prev_track_intances12:Lx4,prev_track_intances13:L"

"$TRT_PATH/bin/trtexec" \
  --onnx=UniAD/onnx/base_e2e_lidar_plan_mapfuse_trt_epoch6_originshift.repaired.onnx \
  --saveEngine=UniAD/engine/base_e2e_lidar_plan_mapfuse_trt_epoch6_originshift_barefp16.engine \
  --staticPlugins=inference_app/enqueueV3/build/libuniad_plugin.so \
  --fp16 \
  --minShapes=${TRACK_SHAPES//L/${MIN}} \
  --optShapes=${TRACK_SHAPES//L/${OPT}} \
  --maxShapes=${TRACK_SHAPES//L/${MAX}} \
  --skipInference \
  2>&1 | tee UniAD/logs/trtexec_base_e2e_lidar_plan_mapfuse_trt_epoch6_originshift_barefp16.log
```

### 5.3 Runtime 命令

```bash
TRT_PATH=/home/baojiali/Downloads/TensorRT-10.7.0.23
export LD_LIBRARY_PATH="$TRT_PATH/lib:${LD_LIBRARY_PATH-}"
OUT=UniAD_train/UniAD/output/base_e2e_lidar_plan_mapfuse_epoch6_10f_originshift_barefp16_20260706
LOG=UniAD/logs/uniad_lidar_e2e_plan_mapfuse_epoch6_10f_originshift_barefp16_20260706.log

rm -rf "$OUT"
inference_app/sparse_lidar/build/uniad_lidar_e2e \
  UniAD_train/UniAD/onnx/base_e2e_lidar_plan_mapfuse_sparse_encoder_epoch6.onnx \
  UniAD/engine/base_e2e_lidar_plan_mapfuse_backbone_neck_epoch6.engine \
  UniAD/engine/base_e2e_lidar_plan_mapfuse_trt_epoch6_originshift_barefp16.engine \
  inference_app/enqueueV3/build/libuniad_plugin.so \
  UniAD_train/UniAD/dumped_inputs/base_e2e_lidar_plan_mapfuse_deploy_data_10f \
  "$OUT" 10 \
  --metadata-json UniAD_train/UniAD/dumped_inputs/base_e2e_lidar_plan_mapfuse_deploy_data_10f \
  --gt-detections UniAD_train/UniAD/dumped_inputs/base_e2e_lidar_plan_mapfuse_deploy_data_10f \
  --track-init-dir UniAD/dumped_inputs/base_e2e_lidar_plan_mapfuse_trt_trace_epoch6_originshift \
  --track-state-len 601 \
  --max-track-state-len 1201 \
  --command 2 \
  --score-threshold 0.2 \
  --bev-max-draw 200 \
  2>&1 | tee "$LOG"
```

## 6. 验证结果

### 6.1 origin-shift bare-FP16 输出

```text
output:
  UniAD_train/UniAD/output/base_e2e_lidar_plan_mapfuse_epoch6_10f_originshift_barefp16_20260706/
sdc_traj files:
  10
all finite:
  true
global min/max:
  all finite; 10-frame runtime verified
```

### 6.2 和 full FP32 对比

对比对象：

```text
UniAD_train/UniAD/output/base_e2e_lidar_plan_mapfuse_epoch6_10f_full_20260706/
```

对比结果：

```text
vs full-FP32 max abs diff:  0.0958 m
vs full-FP32 mean abs diff: 0.0162 m
```

该差异量级在 10 帧 smoke test 上可接受；后续如果要作为正式性能/精度结论，需要扩大到
更多 scene 和完整指标。

### 6.3 可视化输出

统一可视化输出位于：

```text
UniAD_train/UniAD/output/base_e2e_lidar_plan_mapfuse_epoch6_10f_originshift_barefp16_20260706/
```

保留 6 个固定画布 WebM：

```text
base_e2e_lidar_plan_mapfuse_10f_epoch6_originshift_barefp16_20260706_pred_fixed.webm
base_e2e_lidar_plan_mapfuse_10f_epoch6_originshift_barefp16_20260706_compare_fixed.webm
base_e2e_lidar_plan_mapfuse_10f_epoch6_originshift_barefp16_20260706_motion_fixed.webm
base_e2e_lidar_plan_mapfuse_10f_epoch6_originshift_barefp16_20260706_motion_compare_fixed.webm
base_e2e_lidar_plan_mapfuse_10f_epoch6_originshift_barefp16_20260706_planning_fixed.webm
base_e2e_lidar_plan_mapfuse_10f_epoch6_originshift_barefp16_20260706_all_fixed.webm
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

### 8.2 ONNX opset 17 / INormalizationLayer（非当前主线）

TensorRT 日志提示 LayerNorm FP16 reduce/pow 可能溢出。后续可以尝试更高 ONNX opset
或导出成 TensorRT 更容易识别的 normalization 层。不过本次 root-cause 已定位为 map
坐标抵消，origin-shift 裸 FP16 已通过 10 帧验证；normalization 优化不再是当前部署主线。

### 8.3 增加更多验证数据

当前验证是 10 帧 smoke test。正式采用前建议扩大验证：

```text
1. 多 scene runtime
2. full FP32 vs origin-shift bare-FP16 trajectory diff 分布
3. planning L2 / collision 等指标（如果数据中有 planning GT）
4. runtime latency / memory 对比
```

### 8.4 保留 full FP32 fallback

当前保留：

```text
UniAD/engine/base_e2e_lidar_plan_mapfuse_trt_epoch6_fp32.engine
```

如果后续发现 origin-shift bare-FP16 在更大数据集上仍有个别 NaN 或指标异常，可以回退到
full FP32 dense engine 继续部署验证。mapisland69 / attnfp32 已作为历史中间产物清理；
需要复现时再按第 9 节方法重新生成。

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

历史实验曾使用 `tools/build_map_island_engine.sh` 生成 69 节点 FP32 island engine。该 engine、
fp32 spec、旧 fp16safe ONNX 和对应 runtime output 已在 2026-07-07 清理；如需复现，需要
重新导出旧 ONNX、重新生成 spec，并重建 engine。

历史实验中，trtexec 全部 69 层 `Set layer ... to precision fp32` 接受，build PASSED。
当时的 runtime 命令与第 5.3 节一致，仅把 dense engine 换成 `..._mapisland69.engine`。
当时输出目录如下，当前本地已清理：

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

## 10. 治本修复：origin-shift 消除全局坐标 FP16 抵消（推荐首选）

第 9 节把 NaN 定位到 map 分支，但用的仍是"FP32 兜底"这类数值创可贴。进一步追问"map
分支到底哪一步在 FP16 下坏掉"，得到根因并做了治本修复。

### 10.1 根因：全局坐标的灾难性抵消

`MapLaneEncoderTRT._transform_to_ego` 把 HD-map 点从全局坐标系变换到 ego 系：

```python
return (points - trans2).matmul(rot2)
```

`points`（map 点全局坐标）和 `trans2`（ego 全局平移）都是 ~2000-3340 m 量级。两个
~3000 的数相减得到 ego 系 ~百米内的局部坐标，这是**灾难性抵消**：FP16 在 3000 量级的
ULP ~2 m，操作数各自已被量化到 ±2 m 格点，相减结果带米级噪声。实测：

```text
map 坐标范围:                1995.7 .. 3340.6 m   (不溢出 FP16 65504，排除 inf 假设)
(points - ego) FP16 误差:    max 1.67 m   mean 0.52 m   (含旋转)
FP16 在 ~3000 处 ULP:        ~2 m
```

劣化的 ego 系坐标接着喂进 TopK / 归一化 / map cross-attention，误差被放大最终使 sdc_traj
变 NaN。**这不是 attention 本身不稳，是它的输入在 FP16 下已经坏了。** 排除项：
`pos2posemb2d` 的 `dim_t` 范围 1..8660，不溢出，非主因。

### 10.2 修复：常数参考原点（math-identity）

在 `map_lane_encoder.py` 引入常数原点 `o`（map 质心），利用恒等式：

```text
(p - t) == (p - o) - (t - o)
```

- `p - o`（map 点侧，常数）在 `__init__` 用 float64 精确算好、烧进 buffer；
- `t - o`（ego 侧，2 元素）在运行时先以 FP32 计算再 cast，不在 FP16 做大数相减；
- 进图张量从 ~3000 m 缩到 ~144 m，FP16 ULP 相应从 ~2 m 降到 ~0.06 m。

改动文件（部署树；训练树 `MapLaneEncoder` 用 numpy float64 主机端变换，天然精确，无需
改）：

```text
UniAD/projects/mmdet3d_plugin/uniad/dense_heads/motion_head_plugin/map_lane_encoder.py
  __init__:          注册 map_origin buffer，map_central/left/right 存 shift 后坐标
  _transform_to_ego: (trans - origin) 在 fp32 域计算后再 cast
```

### 10.3 数值验证（不需 GPU）

```text
[fp32] 新旧变换 max/mean 误差 = 0.000e+00   (bit 级恒等，不影响精度、无需重训)
[fp16] 旧路径 vs fp32-truth:  max 1.67  mean 0.52  m
[fp16] 新路径 vs fp32-truth:  max 0.19  mean 0.029 m   (mean 改善 ~18x)
```

checkpoint 只含学习权重（`point_mlp.*`/`lane_proj.*`），不含 map buffer，故 `load_state_dict`
不会覆盖 shift；新增 `map_origin` 与既有 `map_central` 同为 map 文件构建的 buffer，行为一致。

### 10.4 端到端验证：裸 FP16、零 FP32 约束

重新导出 ONNX（docker `uniad_torch1.12`，命令同 12.2 节，`--onnx-file` 换
`..._originshift.onnx`）。新 ONNX 中 `map_central` 已 shift（|val| ≤ 557，原 ~3340），
`map_origin` buffer 存在（~2185-2934）。裸 `--fp16` engine build PASSED，10 帧 runtime：

```text
engine                          fp32层数  finite  vs full-fp32 max/mean(m)
full-fp32                       全部       ✓       基准
attnfp32 (1101, 历史大锤)        1101       ✓       0.0912 / 0.0165
mapisland69 (69, 创可贴)         69         ✓       0.0676 / 0.0141
originshift bare-fp16 (治本)     0          ✓       0.0958 / 0.0162
```

治本版 vs mapisland69 仅 max 0.051 / mean 0.006 m，本质同一轨迹。**裸 FP16 全 finite 直接
反证根因**：强制 FP32 不再必要，问题在坐标不在精度模式。

产物：

```text
ONNX:   UniAD/onnx/base_e2e_lidar_plan_mapfuse_trt_epoch6_originshift.repaired.onnx
engine: UniAD/engine/base_e2e_lidar_plan_mapfuse_trt_epoch6_originshift_barefp16.engine
output: UniAD_train/UniAD/output/base_e2e_lidar_plan_mapfuse_epoch6_10f_originshift_barefp16_20260706/
```

### 10.5 建议与边界

- **首选 origin-shift 裸 FP16**：无精度约束、engine 最简单、对 Orin 迁移最友好（不必维护绑定
  ONNX 的 69 节点 spec，无 `--precisionConstraints` 跨平台不确定性）。
- mapisland69 / attnfp32 已降为历史实验记录，本地中间产物已清理；如需复现，再重新生成
  旧 ONNX、FP32 spec 和对应 engine。
- 诚实边界：① 仍为 10 帧 smoke test，非正式精度指标；② 未逐点用 debug tensor 抓首个 NaN
  节点——但裸 FP16 通过等于反证根因，不必再抓；③ Orin 上仍需重建并用实际 bin 复核 finite。

## 11. 同型第二处：velo_update_trt 的全局坐标 FP16 抵消

同一类"全局坐标进 traced op → FP16 灾难性抵消"的隐患在 track 分支也存在，位于
`velo_update_trt`（跨帧传播 track reference points）：

```python
ref_pts = reference_points @ l2g_r1 + l2g_t1 - l2g_t2   # 朴素式
```

连续帧（changed=0）时 `l2g_t1`/`l2g_t2` 是**绝对 ego 全局平移**，实测 nuScenes 部署数据里
约 **2700–3500 m**。先加 `l2g_t1`（抬到 ~3000 m）再减同量级的 `l2g_t2` 就是灾难性抵消，与
`MapLaneEncoderTRT` 同因。

### 11.1 修复

在 fp32 域先算帧间平移差再加，数学恒等 `(a@R + t1) - t2 == a@R + (t1 - t2)`：

```python
trans_delta = (l2g_t1.float() - l2g_t2.float()).to(dtype=ref_pts.dtype)
ref_pts = reference_points @ l2g_r1 + trans_delta
```

连续帧 `l2g_t1 - l2g_t2` 只有 ~0.88 m，进图张量不再出现 ~3000 m 中间量。改动落在部署树两条
TRT 路径：`uniad_track_lidar.py`（LiDAR）与 `uniad_track.py`（camera）的 `velo_update_trt`；
非-trt 的 plain `velo_update`（PyTorch fp32/fp64）保留朴素式，不进 FP16。

### 11.2 验证（TRT-free 隔离）

用真实 dump 的 ref_pts + 真实连续帧 l2g（|t|~3503 m，|t1-t2|~0.88 m）复刻新旧公式：

```text
fp64 new-vs-old 恒等: max|Δ| = 2.5e-13     （数学等价，不影响精度、无需重训）
fp16 OLD 误差 vs truth: mean 0.5715 m  max 2.27 m
fp16 NEW 误差 vs truth: mean 0.0050 m  max 0.057 m   （mean 改善 ~114x）
```

层精度实测（`--exportLayerInfo`）确认本机 SM89 上 OLD engine 的 `MatMul_885`（`@ l2g_r1`）
输入/输出/tactic 全是 `Half`——**是真 FP16，不是 fp32 兜底**，抵消在本机真实存在。

> 注意：当前 10 帧 smoke 部署数据是空 track 种子（obj_idx 全 -1）+ 每帧 reset，
> `velo_update_trt` 不会被触发，因此端到端 A/B 看不到差异；改善由上面隔离测试证明。
> 要端到端复现需构造非空活跃轨迹 + 连续帧 `use_prev_bev=1` 的输入。

工具：`tools/velo_update_isolation.py`（隔离测试）、`tools/compare_track_ab.py`（严格 A/B，
按 dtype 区分 int/float）。提交：子模块 UniAD `da7c5b8`，外层 `09db78f`。

## 12. 全路径裸 FP16 finite 验证矩阵

含 velo_update 修复的代码，对每条 tracking 部署路径重导出 ONNX → 建裸 `--fp16` engine →
跑 10 帧 → 检查所有**浮点**输出 finite（int32 track-state 张量如 obj_idxes/labels/bbox_index
排除；其 -1 sentinel 按 float32 误读会假报 nan）。

```text
环境: SM89 + TensorRT 10.7.0.23, 2026-07-07, 全部裸 --fp16
路径            浮点输出文件   非有限    结论
track_lidar     120           0         ALL_FINITE
drivable        130           0         ALL_FINITE
e2e_lidar       190           0         ALL_FINITE
e2e_occ         200           0         ALL_FINITE
plan-mapfuse    (第 10 节，origin-shift 裸 FP16)   ALL_FINITE
bevformer       无 tracking / 无 velo_update，不受影响，未验
```

脚本：`tools/verify_fp16_all_paths.sh`（`RUN_ONLY="track drivable e2e occ"` 选路径）。

**这一矩阵证明的是**：修复后代码在所有路径都能重导出→建 FP16 engine→跑通、无 NaN/inf
（无回归、没弄坏任何路径）。**不是**端到端展示 velo_update 改善（smoke 数据未触发该路径，
改善见第 11.2 节隔离测试）。Orin/DriveOS 上因 tactic 不同，finite 仍需在目标端重建 + 用
实际 bin 复核。

### 备注：反复出现的 int32-当-float32 假阳性

本会话三次踩同一坑：把 int32 输出（obj_idxes/labels/bbox_index/track_instances 3/4/5/6/13/
max_obj_id）按 float32 读，其 -1 sentinel 的 bit 模式被误认成 nan 或巨大差异。finite 检查与
A/B diff 必须按 dtype 区分，见 `tools/compare_track_ab.py` 的 `INT_KEYS`。
