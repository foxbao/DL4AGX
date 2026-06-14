# base_e2e_lidar_plan.py Planning-Only 部署记录

本文档记录 `UniAD_train/UniAD/projects/configs/stage2_e2e_lidar/base_e2e_lidar_plan.py`
的第一阶段部署方案。当前目标是先把 **tracking + motion + planning** 的
TensorRT 边界跑通，暂时不把 OccHead 接入 dense engine。

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

当前机器上已经有 `base_e2e_lidar` epoch checkpoint，但还没有
`base_e2e_lidar_occ/latest.pth` 和 `base_e2e_lidar_plan/latest.pth`。因此现在可以
完成部署代码和静态验证，但不能做可信的 plan 数值验证。

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
即 keep forward。

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
  UniAD/output/base_e2e_lidar_plan_${TAG}_${NUM_FRAMES}f \
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
  UniAD/output/base_e2e_lidar_plan_${TAG}_${NUM_FRAMES}f \
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

本次已完成：

```text
python3 -m py_compile ...
cmake --build inference_app/sparse_lidar/build -j$(nproc)
```

C++ runtime 编译通过。编译日志里只有 TensorRT 10.7 header 的 deprecation warning。

另外做了一次 smoke 验证。注意：这里临时使用
`base_e2e_lidar/epoch_4.pth` 加载 plan TRT config，因此 planning head 是随机初始化
或 missing 权重，只验证导出 / TensorRT / runtime 链路，不代表真实规划质量。

```text
UniAD/onnx/base_e2e_lidar_plan_trt_smoke.repaired.onnx
UniAD/engine/base_e2e_lidar_plan_trt_smoke.engine
UniAD/output/base_e2e_lidar_plan_smoke_1f/frame_000000_sdc_traj.bin
UniAD/output/base_e2e_lidar_plan_smoke_1f/frame_000000_motion.svg
UniAD/output/base_e2e_lidar_plan_smoke_1f/frame_000000_motion_compare.svg
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

仍未完成的是：使用真正的 `base_e2e_lidar_plan/latest.pth` 做数值验证。原因是当前还
没有 `base_e2e_lidar_plan` checkpoint。
