# base_e2e_lidar_plan.py Planning-Only 部署记录

本文档记录 `UniAD_train/UniAD/projects/configs/stage2_e2e_lidar/base_e2e_lidar_plan.py`
的第一阶段部署方案。当前目标是先把 **tracking + motion + planning** 的
TensorRT 边界跑通，暂时不把 OccHead 接入 dense engine。

runtime 输出统一放在 `UniAD_train/UniAD/output/`；历史上生成在 `UniAD/output/`
下的目录已经剪切到该位置。

## 1. 结论

`base_e2e_lidar_plan.py` 继承链是：

```text
base_track_lidar.py
  -> base_e2e_lidar.py
  -> base_e2e_lidar_occ.py
  -> base_e2e_lidar_plan.py
```

planning head 的关键输入不是 occupancy，而是 MotionHead 里产生的 SDC query：

```text
bev_embed
bev_pos
sdc_traj_query
sdc_track_query
command
```

当前配置里 `use_col_optim=False`，所以 `PlanningHeadSingleMode` 的 TRT forward
不会使用 `occ_mask` 做碰撞后处理。因此第一阶段可以不接 `OccHeadTRTP`，只输出
ego 未来 6 步规划轨迹：

```text
sdc_traj: 1 x 6 x 2
```

## 2. 本次新增部署文件

Python / config：

```text
UniAD/projects/configs/stage2_e2e_lidar/base_e2e_lidar_plan_trt_p.py
UniAD/tools/export_e2e_lidar_plan_onnx.py
```

部署侧模型改动：

```text
UniAD/projects/mmdet3d_plugin/uniad/detectors/uniad_motion_lidar.py
UniAD/projects/mmdet3d_plugin/uniad/dense_heads/motion_head.py
UniAD/projects/mmdet3d_plugin/uniad/dense_heads/__init__.py
```

C++ runtime / 数据 / 可视化改动：

```text
inference_app/sparse_lidar/include/lidar_runtime.hpp
inference_app/sparse_lidar/include/lidar_metadata.hpp
inference_app/sparse_lidar/src/lidar_runtime.cpp
inference_app/sparse_lidar/src/lidar_metadata.cpp
inference_app/sparse_lidar/src/uniad_lidar_e2e.cpp
inference_app/sparse_lidar/visualize_e2e_motion.py
UniAD_train/UniAD/tools/prepare_bevformer_lidar_deploy_data.py
```

## 3. 当前 checkpoint 状态

`base_e2e_lidar_plan.py` 的训练依赖是：

```text
base_e2e_lidar/latest.pth
  -> base_e2e_lidar_occ/latest.pth
  -> base_e2e_lidar_plan/latest.pth
```

当前机器上已经有 `base_e2e_lidar_plan/epoch_1.pth`。2026-07-06 已使用该
checkpoint 完成同 checkpoint 的 sparse encoder、backbone+neck、dense plan engine
部署，并跑通 10 帧 runtime，见第 11 节。

## 4. 准备部署数据

plan runtime 至少需要逐帧 `command`。`prepare_bevformer_lidar_deploy_data.py`
现在会在 sample 里存在这些字段时，把它们写进当前帧 metadata：

```text
command
sdc_planning
sdc_planning_mask
```

示例：

```bash
cd UniAD_train/UniAD

NUM_FRAMES=10

python tools/prepare_bevformer_lidar_deploy_data.py \
  --config projects/configs/stage2_e2e_lidar/base_e2e_lidar_plan.py \
  --split test \
  --start-index 0 \
  --max-frames $NUM_FRAMES \
  --out-dir dumped_inputs/base_e2e_lidar_plan_deploy_data_${NUM_FRAMES}f
```

如果 metadata 中没有 `command`，C++ runtime 会使用命令行默认值 `--command 2`
即 keep forward。2026-07-06 生成的 10 帧部署数据中，dataset sample 没有提供
top-level `command/sdc_planning/sdc_planning_mask`，所以本次 runtime 显式使用
`--command 2`。

## 5. 导出 Dense Plan ONNX

等 `base_e2e_lidar_plan` checkpoint 训练出来后：

```bash
cd UniAD

TAG=plan_epoch2
CKPT=../UniAD_train/UniAD/projects/work_dirs/stage2_e2e_lidar/base_e2e_lidar_plan/latest.pth

CUDA_VISIBLE_DEVICES=0 python tools/export_e2e_lidar_plan_onnx.py \
  projects/configs/stage2_e2e_lidar/base_e2e_lidar_plan_trt_p.py \
  $CKPT \
  --onnx-file onnx/base_e2e_lidar_plan_trt_${TAG}.onnx \
  --track-state-len 601 \
  --command 2 \
  --dump-input-dir dumped_inputs/base_e2e_lidar_plan_trt_trace_${TAG}

cd -
```

输出：

```text
UniAD/onnx/base_e2e_lidar_plan_trt_${TAG}.onnx
UniAD/onnx/base_e2e_lidar_plan_trt_${TAG}.repaired.onnx
UniAD/dumped_inputs/base_e2e_lidar_plan_trt_trace_${TAG}/*.dat
```

与 `base_e2e_lidar_trt_p.py` 相比，plan ONNX 多一个输入和一个输出：

```text
input:  command  shape=1
output: sdc_traj shape=1x6x2
```

## 6. 编译 TensorRT Engine

Backbone+neck engine 继续复用 `base_e2e_lidar` 文档里的导出方式。Dense plan engine：

```bash
TRT_PATH=/home/baojiali/Downloads/TensorRT-10.7.0.23

MIN=601
OPT=601
MAX=1201

TRACK_SHAPES=prev_track_intances0:Lx512,prev_track_intances1:Lx3,prev_track_intances3:L,prev_track_intances4:L,prev_track_intances5:L,prev_track_intances6:L,prev_track_intances8:L,prev_track_intances9:Lx10,prev_track_intances11:Lx4x256,prev_track_intances12:Lx4,prev_track_intances13:L

$TRT_PATH/bin/trtexec \
  --onnx=UniAD/onnx/base_e2e_lidar_plan_trt_${TAG}.repaired.onnx \
  --saveEngine=UniAD/engine/base_e2e_lidar_plan_trt_${TAG}.engine \
  --staticPlugins=inference_app/enqueueV3/build/libuniad_plugin.so \
  --fp16 \
  --minShapes=${TRACK_SHAPES//L/${MIN}} \
  --optShapes=${TRACK_SHAPES//L/${OPT}} \
  --maxShapes=${TRACK_SHAPES//L/${MAX}} \
  --skipInference \
  2>&1 | tee UniAD/logs/trtexec_base_e2e_lidar_plan_trt_${TAG}.log
```

固定输入包括：

```text
lidar_bev:      1x256x120x160
prev_bev:       1x256x120x160
shift:          1x2
use_prev_bev:   1
prev_timestamp: 1
timestamp:      1
prev_l2g_r_mat: 1x3x3
l2g_r_mat:      1x3x3
prev_l2g_t:     1x3
l2g_t:          1x3
max_obj_id:     1
command:        1
```

## 7. C++ Runtime

`uniad_lidar_e2e` 已经做成兼容旧 engine 和 plan engine：

- 如果 engine 没有 `command` 输入，不启用 planning。
- 如果 engine 有 `command` 输入，runtime 从 metadata 读取逐帧 command。
- 如果 metadata 没有 command，使用 `--command`，默认 `2`。
- 如果 engine 有 `sdc_traj` 输出，runtime 会写 `frame_XXXXXX_sdc_traj.bin`。

运行示例：

```bash
export LD_LIBRARY_PATH=$TRT_PATH/lib:$LD_LIBRARY_PATH

NUM_FRAMES=10

inference_app/sparse_lidar/build/uniad_lidar_e2e \
  UniAD_train/UniAD/onnx/base_e2e_lidar_sparse_encoder_${TAG}.onnx \
  UniAD/engine/base_e2e_lidar_backbone_neck_${TAG}.engine \
  UniAD/engine/base_e2e_lidar_plan_trt_${TAG}.engine \
  inference_app/enqueueV3/build/libuniad_plugin.so \
  UniAD_train/UniAD/dumped_inputs/base_e2e_lidar_plan_deploy_data_${NUM_FRAMES}f \
  UniAD_train/UniAD/output/base_e2e_lidar_plan_${TAG}_${NUM_FRAMES}f \
  $NUM_FRAMES \
  --metadata-json UniAD_train/UniAD/dumped_inputs/base_e2e_lidar_plan_deploy_data_${NUM_FRAMES}f \
  --gt-detections UniAD_train/UniAD/dumped_inputs/base_e2e_lidar_plan_deploy_data_${NUM_FRAMES}f \
  --track-init-dir UniAD/dumped_inputs/base_e2e_lidar_plan_trt_trace_${TAG} \
  --track-state-len 601 \
  --max-track-state-len 1201 \
  --command 2 \
  --score-threshold 0.2 \
  --bev-max-draw 200 \
  2>&1 | tee UniAD/logs/uniad_lidar_e2e_plan_${TAG}_${NUM_FRAMES}f.log
```

## 8. 可视化

`visualize_e2e_motion.py` 会自动检测 `frame_XXXXXX_sdc_traj.bin`。如果存在，
它会在 motion SVG 里画出 ego planning 轨迹。

```bash
python3 inference_app/sparse_lidar/visualize_e2e_motion.py \
  UniAD_train/UniAD/output/base_e2e_lidar_plan_${TAG}_${NUM_FRAMES}f \
  --gt-dir UniAD_train/UniAD/dumped_inputs/base_e2e_lidar_plan_deploy_data_${NUM_FRAMES}f \
  --num-frames $NUM_FRAMES \
  --top-modes 3 \
  --min-score 0.2
```

输出示例：

```text
frame_000000_motion.svg
frame_000000_motion_compare.svg
```

## 9. 后续完整 occ + plan

完整对齐 `base_e2e_lidar_plan.py` 还需要接入 `OccHeadTRTP`：

- 输出 `seg_out`
- 把 `track_query`、`track_query_pos`、`traj_query`、`track_scores` 传给 OccHead
- 根据 `occ_n_future=6` 确认 `seg_out` 的时间维
- 如果后续打开 `use_col_optim=True`，planning head 才会真正依赖 `occ_mask`

当前第一阶段先不做这部分，避免在没有 plan checkpoint 时扩大变量。

## 10. 已完成验证

早期已完成：

```text
python3 -m py_compile ...
cmake --build inference_app/sparse_lidar/build -j$(nproc)
```

C++ runtime 编译通过。编译日志里只有 TensorRT 10.7 header 的 deprecation warning。

另外做过一次 smoke 验证。注意：这里临时使用
`base_e2e_lidar/epoch_4.pth` 加载 plan TRT config，因此 planning head 是随机初始化
或 missing 权重，只验证导出 / TensorRT / runtime 链路，不代表真实规划质量。

```text
UniAD/onnx/base_e2e_lidar_plan_trt_smoke.repaired.onnx
UniAD/engine/base_e2e_lidar_plan_trt_smoke.engine
UniAD_train/UniAD/output/base_e2e_lidar_plan_smoke_1f/frame_000000_sdc_traj.bin
UniAD_train/UniAD/output/base_e2e_lidar_plan_smoke_1f/frame_000000_motion.svg
UniAD_train/UniAD/output/base_e2e_lidar_plan_smoke_1f/frame_000000_motion_compare.svg
```

smoke 结果：

```text
PyTorch output sdc_traj: 1 x 6 x 2
TensorRT 10.7 trtexec --skipInference: PASSED
TensorRT network: 23 inputs, 29 outputs
command input: 1 float32
sdc_traj output: 1 x 6 x 2 float32
uniad_lidar_e2e smoke runtime: 1 frame passed, 4 tracks, max_obj_id=4
```

真实 `base_e2e_lidar_plan/epoch_1.pth` 的部署记录见下一节。

## 11. 2026-07-06 epoch1 实际部署结果

本次部署使用：

```text
config:     UniAD_train/UniAD/projects/configs/stage2_e2e_lidar/base_e2e_lidar_plan.py
checkpoint: UniAD_train/UniAD/projects/work_dirs/stage2_e2e_lidar/base_e2e_lidar_plan/epoch_1.pth
command:    2
frames:     10
TensorRT:   10.7.0
GPU:        RTX 4090, SM 8.9
```

### 11.1 导出产物

```text
UniAD_train/UniAD/dumped_inputs/base_e2e_lidar_plan_deploy_data_10f/
UniAD_train/UniAD/onnx/base_e2e_lidar_plan_sparse_encoder_epoch1.onnx
UniAD_train/UniAD/onnx/base_e2e_lidar_plan_backbone_neck_epoch1.onnx
UniAD/onnx/base_e2e_lidar_plan_trt_epoch1.onnx
UniAD/onnx/base_e2e_lidar_plan_trt_epoch1.repaired.onnx
UniAD/dumped_inputs/base_e2e_lidar_plan_trt_trace_epoch1/
```

Dense ONNX trace 输出：

```text
sdc_traj: (1, 6, 2)
```

Dense 导出边界不包含 sparse encoder / backbone / neck / seg head，所以加载 checkpoint
时出现 frontend 和 `seg_head.*` 相关的 `unexpected key` 警告属于预期。

### 11.2 TensorRT engine

```text
UniAD/engine/base_e2e_lidar_plan_backbone_neck_epoch1.engine
UniAD/engine/base_e2e_lidar_plan_trt_epoch1.engine
```

`trtexec --skipInference` 已通过。Dense plan engine 信息：

```text
inputs:  23
outputs: 29
extra input:  command, shape=1, float32
extra output: sdc_traj, shape=1x6x2, float32
```

### 11.3 10 帧 runtime

运行日志和输出目录：

```text
UniAD/logs/uniad_lidar_e2e_plan_epoch1_10f_full_20260706.log
UniAD_train/UniAD/output/base_e2e_lidar_plan_epoch1_10f_full_20260706/
```

输出检查：

```text
detections_txt:       10
bev_svg:              10
bev_compare_svg:      10
valid_traj_masks_bin: 10
sdc_traj_bin:         10
```

`sdc_traj.bin` layout：

```text
float32[1, 6, 2]
```

抽查数值：

```text
frame_000000_sdc_traj.bin first=[0.6665, -0.0571], last=[3.4023, 0.0946]
frame_000001_sdc_traj.bin first=[0.8237, -0.0711], last=[4.1797, 0.0349]
frame_000005_sdc_traj.bin last=[4.7109, -0.0110]
global finite: true
global min/max: -0.0729 / 4.7109
```

### 11.4 可视化

`visualize_e2e_motion.py` 曾生成旧版 motion SVG：

```text
UniAD_train/UniAD/output/base_e2e_lidar_plan_epoch1_10f_full_20260706/frame_000000_motion.svg
UniAD_train/UniAD/output/base_e2e_lidar_plan_epoch1_10f_full_20260706/frame_000000_motion_compare.svg
```

另外曾生成更直观的 PNG / WebM overlay；上述旧版 SVG、PNG 中间帧和 overlay WebM
已经清理。当前保留并推荐查看的是第
12.5 节统一工具生成的 `base_e2e_lidar_plan_10f_epoch1_full_20260706_*_fixed.webm`。

overlay 约定：

```text
gray  = 当前帧 LiDAR points
green = GT detection boxes
blue  = engine 输出的 ego planning sdc_traj
```

抽查 `frame_000000` 和 `frame_000005`，蓝色 ego planning 轨迹从自车原点向 x 正方向
平滑前进，没有 NaN、离谱跳变或坐标翻转。由于本次部署数据没有 `sdc_planning`
GT，当前只能做定性可视化，还不能计算 planning L2。

## 12. 2026-07-06 mapfuse epoch6 实际部署结果

本次部署目标：

```text
config:     UniAD_train/UniAD/projects/configs/stage2_e2e_lidar/base_e2e_lidar_plan_mapfuse.py
checkpoint: UniAD_train/UniAD/projects/work_dirs/stage2_e2e_lidar/base_e2e_lidar_plan_mapfuse/epoch_6.pth
command:    2
frames:     10
TensorRT:   10.7.0
GPU:        RTX 4090, SM 8.9
```

`base_e2e_lidar_plan_mapfuse.py` 相比 `base_e2e_lidar_plan.py` 额外打开了：

```text
map_lane_encoder.map_path=data/kl_8/map/base_map.txt
map_lane_encoder.num_lanes=64
map_lane_encoder.num_points_per_lane=20
motion_head.map_agent_scope=none
planning_head.use_map_lane=True
planning_head.map_local_k=16
planning_head.map_attn_layers=1
planning_head.map_gate_init=-2.0
```

当前 TRT 适配只把 HD-map lane feature 接入 ego planning branch；
actor motion 仍按训练配置 `map_agent_scope=none`，不使用 map lane。

### 12.1 代码适配点

新增 / 修改：

```text
UniAD/projects/mmdet3d_plugin/uniad/dense_heads/motion_head_plugin/map_lane_encoder.py
UniAD/projects/configs/stage2_e2e_lidar/base_e2e_lidar_plan_mapfuse_trt_p.py
UniAD/projects/mmdet3d_plugin/uniad/dense_heads/planning_head.py
UniAD/projects/mmdet3d_plugin/uniad/detectors/uniad_motion_lidar.py
```

关键点：

- `MapLaneEncoderTRT` 在模型构建 / ONNX export 阶段解析 `base_map.txt`，并把 map lane
  arrays 注册为 buffer；engine runtime 不再依赖文本 map 文件。
- `MapLaneEncoderTRT.forward()` 用当前帧 `l2g_r_mat/l2g_t` 把全局 map lane 转到 ego
  坐标系，筛选 BEV 范围内距离最近的 64 条 lane。
- 为避免 TensorRT parser 遇到 `org.pytorch.aten::inverse`，map 坐标变换没有使用
  `torch.inverse(4x4)`，而是显式使用二维刚体逆变换：

```text
(global_xy - trans_xy) @ rot_xy
```

该替换与原 4x4 inverse 写法在 FP32 下最大差异约 `1.5e-05`。

### 12.2 导出命令

部署数据：

```bash
cd UniAD_train/UniAD
python tools/prepare_bevformer_lidar_deploy_data.py \
  --config projects/configs/stage2_e2e_lidar/base_e2e_lidar_plan_mapfuse.py \
  --split test \
  --start-index 0 \
  --max-frames 10 \
  --out-dir dumped_inputs/base_e2e_lidar_plan_mapfuse_deploy_data_10f \
  2>&1 | tee logs/prepare_base_e2e_lidar_plan_mapfuse_deploy_data_10f.log
```

Sparse encoder / backbone-neck ONNX：

```text
UniAD_train/UniAD/onnx/base_e2e_lidar_plan_mapfuse_sparse_encoder_epoch6.onnx
UniAD_train/UniAD/onnx/base_e2e_lidar_plan_mapfuse_backbone_neck_epoch6.onnx
```

Dense plan ONNX：

```bash
cd UniAD
CUDA_VISIBLE_DEVICES=0 python tools/export_e2e_lidar_plan_onnx.py \
  projects/configs/stage2_e2e_lidar/base_e2e_lidar_plan_mapfuse_trt_p.py \
  ../UniAD_train/UniAD/projects/work_dirs/stage2_e2e_lidar/base_e2e_lidar_plan_mapfuse/epoch_6.pth \
  --onnx-file onnx/base_e2e_lidar_plan_mapfuse_trt_epoch6.onnx \
  --track-state-len 601 \
  --command 2 \
  --dump-input-dir dumped_inputs/base_e2e_lidar_plan_mapfuse_trt_trace_epoch6 \
  2>&1 | tee logs/export_e2e_lidar_plan_mapfuse_trt_epoch6.log
```

Dense ONNX trace 输出：

```text
sdc_traj: (1, 6, 2)
```

导出产物：

```text
UniAD_train/UniAD/dumped_inputs/base_e2e_lidar_plan_mapfuse_deploy_data_10f/
UniAD_train/UniAD/onnx/base_e2e_lidar_plan_mapfuse_sparse_encoder_epoch6.onnx
UniAD_train/UniAD/onnx/base_e2e_lidar_plan_mapfuse_backbone_neck_epoch6.onnx
UniAD/onnx/base_e2e_lidar_plan_mapfuse_trt_epoch6.onnx
UniAD/onnx/base_e2e_lidar_plan_mapfuse_trt_epoch6.repaired.onnx
UniAD/dumped_inputs/base_e2e_lidar_plan_mapfuse_trt_trace_epoch6/
```

加载 checkpoint 时出现 frontend、`seg_head.*`、`occ_head.*` 相关 `unexpected key`
属于 dense export boundary 的预期现象；`map_lane_encoder.*` buffer 的 `missing key`
也属于预期，因为这些 buffer 来自部署时解析的静态 map。

### 12.3 TensorRT engine

Backbone-neck engine：

```bash
TRT_PATH=/home/baojiali/Downloads/TensorRT-10.7.0.23
export LD_LIBRARY_PATH="$TRT_PATH/lib:${LD_LIBRARY_PATH-}"

"$TRT_PATH/bin/trtexec" \
  --onnx=UniAD_train/UniAD/onnx/base_e2e_lidar_plan_mapfuse_backbone_neck_epoch6.onnx \
  --saveEngine=UniAD/engine/base_e2e_lidar_plan_mapfuse_backbone_neck_epoch6.engine \
  --fp16 \
  --skipInference \
  2>&1 | tee UniAD/logs/trtexec_base_e2e_lidar_plan_mapfuse_backbone_neck_epoch6.log
```

**（2026-07-06 更新）推荐方案：origin-shift 代码修复 + 裸 `--fp16`。**

此前认为 mapfuse dense plan engine 必须带 FP32 attention island（见下方"历史方案"）。
后续定位发现真正根因是 `MapLaneEncoderTRT._transform_to_ego` 把 HD-map 点从**全局
坐标系**（~2000-3340 m）变换到 ego 系时，在 FP16 下发生**灾难性抵消**（两个 ~3000 的
数相减，FP16 在该量级 ULP ~2 m），劣化的坐标毒化整个 map 分支，而非 attention 本身
不稳。修复方式是在 `map_lane_encoder.py` 中引入常数参考原点（map 质心）：

```text
(p - t) == (p - o) - (t - o)      # o = map 质心常数
```

数学恒等（FP32 下 bit 级不变，不影响精度、无需重训），但进图张量从 ~3000 m 缩到
~144 m，FP16 抵消误差降 ~18 倍（mean 0.52 m -> 0.029 m）。修复后重新导出 ONNX
（`..._originshift.repaired.onnx`），**裸 `--fp16`、零 FP32 约束**即可，`sdc_traj` 全 finite：

```bash
TRT_PATH=/home/baojiali/Downloads/TensorRT-10.7.0.23
export LD_LIBRARY_PATH="$TRT_PATH/lib:${LD_LIBRARY_PATH-}"
MIN=601; OPT=601; MAX=1201
TRACK_SHAPES="prev_track_intances0:Lx512,prev_track_intances1:Lx3,prev_track_intances3:L,prev_track_intances4:L,prev_track_intances5:L,prev_track_intances6:L,prev_track_intances8:L,prev_track_intances9:Lx10,prev_track_intances11:Lx4x256,prev_track_intances12:Lx4,prev_track_intances13:L"

"$TRT_PATH/bin/trtexec" \
  --onnx=UniAD/onnx/base_e2e_lidar_plan_mapfuse_trt_epoch6_originshift.repaired.onnx \
  --saveEngine=UniAD/engine/base_e2e_lidar_plan_mapfuse_trt_epoch6_originshift_barefp16.engine \
  --staticPlugins=inference_app/enqueueV3/build/libuniad_plugin.so \
  --fp16 \
  --minShapes=${TRACK_SHAPES//L/${MIN}} \
  --optShapes=${TRACK_SHAPES//L/${OPT}} \
  --maxShapes=${TRACK_SHAPES//L/${MAX}} \
  --skipInference
```

10 帧 runtime `sdc_traj` 全 finite，vs full-FP32 max/mean abs diff = 0.096 / 0.0162 m
（与下方 attnfp32 大锤同量级）。裸 FP16 对 Orin/DriveOS 迁移最友好：无需维护绑定 ONNX
的 FP32 节点名 spec，也没有 `--precisionConstraints` 跨平台不确定性。详见
`MAPFUSE_FP16_NAN_ANALYSIS.md` 第 10 节。

> 注意：origin-shift 是 export 期烧进 initializer 的改动，必须**重新导出 ONNX** 才生效；
> 直接拿旧的 `..._fp16safe.repaired.onnx` 跑裸 FP16 仍会 NaN。

---

**历史方案（已被 origin-shift 取代，保留作记录）：FP32 attention island 混合精度 engine。**

修复根因前，需在整体 `--fp16` 下把 attention / MLP 的 `MatMul/Gemm/Softmax` 和归一化的
`ReduceMean/Pow/Sqrt/Div` 约束为 FP32（后续定位发现只需 map 分支 69 个节点即可，见
`MAPFUSE_FP16_NAN_ANALYSIS.md` 第 9 节）。此路径仍可用作"不想改模型代码/重导出"时的
备选：

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

注意：裸 `--fp16` 是否可用取决于 ONNX 是否带 origin-shift 修复。**未修复**的旧 ONNX
（`..._fp16safe.repaired.onnx` 及更早）裸 FP16 会 `sdc_traj` 全 NaN（PyTorch dense forward
正常，说明来自 FP16 下 map 分支全局坐标抵消，见本节开头）；**已修复**的
`..._originshift.repaired.onnx` 裸 FP16 全 finite。已删除未修复 ONNX 上误用风险较高的
裸 FP16 dense engine：

```text
UniAD/engine/base_e2e_lidar_plan_mapfuse_trt_epoch6.engine
```

本次还验证了两个中间修复路径，均失败并已清理对应 engine / output：

```text
1. invalid lane mask sentinel: 1.0e8 -> 1.0e4，仍然 sdc_traj 全 NaN
2. 仅 ReduceMean/Pow/Sqrt 强制 FP32，仍然 sdc_traj 全 NaN
```

保留并验证通过的 engine：

```text
UniAD/engine/base_e2e_lidar_plan_mapfuse_backbone_neck_epoch6.engine
UniAD/engine/base_e2e_lidar_plan_mapfuse_trt_epoch6_originshift_barefp16.engine  # 推荐（治本，裸 fp16）
UniAD/engine/base_e2e_lidar_plan_mapfuse_trt_epoch6_mapisland69.engine           # 备选（69 节点 fp32 island，不改代码）
UniAD/engine/base_e2e_lidar_plan_mapfuse_trt_epoch6_fp16safe_attnfp32.engine     # 历史（1101 节点大锤）
UniAD/engine/base_e2e_lidar_plan_mapfuse_trt_epoch6_fp32.engine                  # fallback
```

### 12.4 10 帧 runtime

运行命令：

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

输出目录：

```text
UniAD_train/UniAD/output/base_e2e_lidar_plan_mapfuse_epoch6_10f_attnfp32_20260706/
```

输出检查：

```text
bbox:        10
scores:      20
labels:      10
traj:        20
traj_scores: 10
sdc_traj:    10
motion_svg:  10
compare_svg: 10
```

`sdc_traj.bin` layout：

```text
float32[1, 6, 2]
```

抽查数值：

```text
frame_000000_sdc_traj.bin first=[0.7432, -0.0188], last=[4.3359, -0.1506]
frame_000001_sdc_traj.bin first=[0.8462,  0.0027], last=[5.0781, -0.0136]
frame_000005_sdc_traj.bin first=[0.9360, -0.0082], last=[5.5703, -0.0786]
frame_000009_sdc_traj.bin first=[1.0918,  0.0257], last=[6.6016,  0.1272]
all_sdc_finite: true
global min/max: -0.1506 / 6.6016
vs full-FP32 max abs diff:  0.0912 m
vs full-FP32 mean abs diff: 0.0165 m
```

### 12.5 统一可视化

当前统一使用：

```text
inference_app/sparse_lidar/visualize_e2e_outputs.py
```

这个脚本直接读取 `uniad_lidar_e2e` runtime 输出的 `.bin/.txt` 文件和 deploy data
里的 `raw_points_*.bin`，统一生成固定画布 PNG / WebM。后续不要再混用
Chrome 截 SVG、临时 matplotlib overlay、单独 planning overlay 等多套流程。

统一命令：

```bash
python inference_app/sparse_lidar/visualize_e2e_outputs.py \
  UniAD_train/UniAD/output/base_e2e_lidar_plan_mapfuse_epoch6_10f_attnfp32_20260706 \
  --data-dir UniAD_train/UniAD/dumped_inputs/base_e2e_lidar_plan_mapfuse_deploy_data_10f \
  --gt-dir UniAD_train/UniAD/dumped_inputs/base_e2e_lidar_plan_mapfuse_deploy_data_10f \
  --num-frames 10 \
  --tag base_e2e_lidar_plan_mapfuse_10f_epoch6_attnfp32_20260706 \
  --modes pred compare motion motion_compare planning all \
  --width 1200 \
  --height 900 \
  --fps 2 \
  --min-score 0.2
```

统一输出：

```text
base_e2e_lidar_plan_mapfuse_10f_epoch6_attnfp32_20260706_pred_fixed.webm
base_e2e_lidar_plan_mapfuse_10f_epoch6_attnfp32_20260706_compare_fixed.webm
base_e2e_lidar_plan_mapfuse_10f_epoch6_attnfp32_20260706_motion_fixed.webm
base_e2e_lidar_plan_mapfuse_10f_epoch6_attnfp32_20260706_motion_compare_fixed.webm
base_e2e_lidar_plan_mapfuse_10f_epoch6_attnfp32_20260706_planning_fixed.webm
base_e2e_lidar_plan_mapfuse_10f_epoch6_attnfp32_20260706_all_fixed.webm
```

模式含义：

```text
pred           = LiDAR points + detection / tracking boxes
compare        = pred + GT boxes（如果 deploy data 中存在）
motion         = pred + object motion trajectories
motion_compare = motion + GT boxes（如果 deploy data 中存在）
planning       = LiDAR points + ego planning sdc_traj
all            = LiDAR points + detection boxes + object motion + ego planning
```

中间 PNG 帧用于合成 WebM，合成完成后可以清理；正式查看只保留 `*_fixed.webm`。

所有统一视频均为固定规格：

```text
width:  1200
height: 900
fps:    2
```

实测输出：

```text
all_fixed.webm:            1200x900, 2 fps, size=36K
compare_fixed.webm:        1200x900, 2 fps, size=32K
motion_compare_fixed.webm: 1200x900, 2 fps, size=32K
motion_fixed.webm:         1200x900, 2 fps, size=29K
planning_fixed.webm:       1200x900, 2 fps, size=23K
pred_fixed.webm:           1200x900, 2 fps, size=29K
```

这套固定画布解决了之前不同脚本导致的画面一会儿大一会儿小、部分视频带白边的问题。
当前 deploy data 未带 `sdc_planning` GT，因此 planning 仍只能做定性可视化，不能计算
planning L2。
