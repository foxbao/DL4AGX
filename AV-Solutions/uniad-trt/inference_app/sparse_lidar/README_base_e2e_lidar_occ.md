# base_e2e_lidar_occ.py Occupancy 可视化与 TensorRT 部署

> **FP16 数值卫生（2026-07-07）**：dense engine 继承 `UniADTrackLidarTRT.velo_update_trt`，
> 存在全局坐标 FP16 灾难性抵消隐患，已在源码修复。**engine 需用含修复的代码重新导出**；
> 2026-07-07 之前构建的 engine 不含修复。本机已验证重导出后裸 FP16 10 帧全 finite
> （`tools/verify_fp16_all_paths.sh`）。根因与验证矩阵见 `MAPFUSE_FP16_NAN_ANALYSIS.md`
> 第 11–12 节。

本文档前半部分记录 `base_e2e_lidar_occ.py` 的训练侧 OccHead 可视化方法。它直接
加载 PyTorch checkpoint 做少量 `val/test` 前向，然后画出当前帧和未来帧的
occupancy。

`visualize_e2e_occ.py` 这个脚本不是 TensorRT 部署链路的一部分，目的是快速检查：

- `gt_segmentation` 真值是否正常。
- `seg_out` 是否学到车辆占用区域。
- 哪些网格是 TP / FP / FN。
- LiDAR 点云与 GT / Pred occ 的空间关系是否合理。

2026-07-06 之前，本仓库只有上述 PyTorch 可视化记录，没有
`base_e2e_lidar_occ.py` 的完整 TensorRT 部署说明。本次已补齐 dense E2E+OCC
ONNX 导出、TensorRT engine 构建、C++ 运行时 `seg_out` 输出，以及 10 帧
end-to-end smoke test，见第 6 节。

runtime 输出统一放在 `UniAD_train/UniAD/output/`；历史上生成在 `UniAD/output/`
下的目录已经剪切到该位置。

## 1. 脚本

```text
inference_app/sparse_lidar/visualize_e2e_occ.py
```

输出的一张图只包含一个 occ 时间步：

```text
left  : GT occ
right : Pred occ
```

当前帧 LiDAR 点云会以浅色小点叠加在左右两张图上。未来 `t+0.5s`、
`t+1.0s` 等图里叠加的仍然是当前 sample 的点云，只作为空间参照，不代表未来
点云。

`base_e2e_lidar_occ.py` 当前设置：

```text
occ_n_future = 4
```

所以每个 dataset sample 会生成 5 张连续图片：

```text
t+0.0s, t+0.5s, t+1.0s, t+1.5s, t+2.0s
```

如果 `--max-frames 6`，最终会生成 `6 x 5 = 30` 张 PNG，并按顺序合成
`occ_vis.webm`。

## 2. 运行命令

从仓库根目录执行：

```bash
cd /home/baojiali/Downloads/public_code/DL4AGX/AV-Solutions/uniad-trt
conda activate uniad_train

CUDA_VISIBLE_DEVICES=5 python3 inference_app/sparse_lidar/visualize_e2e_occ.py \
  --config UniAD_train/UniAD/projects/configs/stage2_e2e_lidar/base_e2e_lidar_occ.py \
  --checkpoint UniAD_train/UniAD/projects/work_dirs/stage2_e2e_lidar/base_e2e_lidar_occ/epoch_1.pth \
  --split val \
  --start-index 0 \
  --max-frames 6 \
  --device cuda:0 \
  --save-arrays \
  --point-stride 4 \
  --webm-fps 2 \
  --out-dir UniAD_train/UniAD/projects/work_dirs/vis_base_e2e_lidar_occ_epoch1
```

注意：如果使用 `CUDA_VISIBLE_DEVICES=5`，脚本内部看到的 GPU 是 `cuda:0`。
如果不设置 `CUDA_VISIBLE_DEVICES`，可以直接写 `--device cuda:5`。

点云太密时可以调大：

```bash
--point-stride 8
```

如果只想看 occupancy，不叠点云：

```bash
--no-points
```

## 3. 输出

```text
frame_000000_occ.png
frame_000001_occ.png
summary.json
index.html
occ_vis.webm
```

如果加了 `--save-arrays`，还会输出：

```text
frame_000000_occ.npz
```

里面保存：

```text
seg_gt
seg_out
ins_seg_gt
ins_seg_out
```

`summary.json` 会记录每个时间步的局部指标：

```text
iou
precision
recall
gt_cells
pred_cells
tp / fp / fn
```

这只是可视化子集上的快速检查，不等同于完整验证集里的
`occ_center30m_iou` / `occ_full_iou`。

## 4. 指定样本

可以用 dataset index：

```bash
--start-index 120 --max-frames 4
```

也可以从指定 sample token 或 scene token 开始：

```bash
--token <sample_token>
--scene-token <scene_token>
```

脚本会从起点开始取同一个 scene 内的连续帧，并保持模型的 tracking/prev-BEV
状态。想看稳定的连续预测，最好从 scene 的前几帧开始；从中间帧开始也能跑，
但 tracking 状态没有前文，occ 结果可能更抖。

## 5. 看图建议

优先看三件事：

1. GT 是否在车辆真实位置附近，且未来帧会随车辆运动。
2. Pred 是否覆盖主要车辆区域，不只是当前帧记忆。
3. 图底部的 `TP / FP / FN` 和 `step IoU` 是否随未来时间明显恶化。

如果 `Pred occ` 明显偏空，通常是 threshold 或 occ head 还没收敛；如果明显铺满，
通常是 score/track query 噪声或 `test_seg_thresh` 太低。当前配置里：

```text
test_seg_thresh = 0.1
test_with_track_score = True
```

## 6. TensorRT 部署记录（2026-07-06）

> 本节是标准 LiDAR 部署流程（导出 ONNX → 编译 Engine → C++ Runtime），与
> `README_base_e2e_lidar.md` 一致，仅 dense engine 多一个 `seg_out` 输出。
> 前面第 1–5 节是 PyTorch 侧的 occ 可视化工具，与 TensorRT 部署独立。

本次新增/修改的部署侧文件：

- `UniAD/projects/configs/stage2_e2e_lidar/base_e2e_lidar_occ_trt_p.py`
- `UniAD/tools/export_e2e_lidar_occ_onnx.py`
- `UniAD/projects/mmdet3d_plugin/uniad/detectors/uniad_motion_lidar.py`
- `inference_app/sparse_lidar/src/uniad_lidar_e2e.cpp`

部署使用的 checkpoint：

```text
UniAD_train/UniAD/projects/work_dirs/stage2_e2e_lidar/base_e2e_lidar_occ/epoch_4.pth
```

### 6.1 前端 ONNX 导出

从仓库根目录执行：

```bash
conda activate uniad_train
cd UniAD_train/UniAD
export PYTHONPATH=$(pwd):${PYTHONPATH-}

CFG=projects/configs/stage2_e2e_lidar/base_e2e_lidar_occ.py
CKPT=projects/work_dirs/stage2_e2e_lidar/base_e2e_lidar_occ/epoch_4.pth

CUDA_VISIBLE_DEVICES=0 python tools/export_bevformer_lidar_sparse_onnx.py \
  "$CFG" "$CKPT" --split test --index 0 \
  --onnx-file onnx/base_e2e_lidar_occ_sparse_encoder_epoch4.onnx \
  --tensor-prefix dumped_inputs/base_e2e_lidar_occ_sparse_encoder_epoch4/infer \
  2>&1 | tee logs/export_base_e2e_lidar_occ_sparse_encoder_epoch4.log

CUDA_VISIBLE_DEVICES=0 python tools/export_bevformer_lidar_backbone_neck_onnx.py \
  "$CFG" "$CKPT" \
  --onnx-file onnx/base_e2e_lidar_occ_backbone_neck_epoch4.onnx \
  2>&1 | tee logs/export_base_e2e_lidar_occ_backbone_neck_epoch4.log
```

产物：

```text
UniAD_train/UniAD/onnx/base_e2e_lidar_occ_sparse_encoder_epoch4.onnx
UniAD_train/UniAD/onnx/base_e2e_lidar_occ_backbone_neck_epoch4.onnx
```

前端导出只使用 sparse encoder / backbone / neck，日志里和 occ head 相关的
unused/missing key 警告不影响这两个前端 ONNX。

### 6.2 Dense E2E+OCC ONNX 导出

从 `UniAD/` 执行：

```bash
conda activate uniad_train
cd /home/baojiali/Downloads/public_code/DL4AGX/AV-Solutions/uniad-trt/UniAD
export PYTHONPATH=$(pwd):${PYTHONPATH-}

CUDA_VISIBLE_DEVICES=0 python tools/export_e2e_lidar_occ_onnx.py \
  projects/configs/stage2_e2e_lidar/base_e2e_lidar_occ_trt_p.py \
  ../UniAD_train/UniAD/projects/work_dirs/stage2_e2e_lidar/base_e2e_lidar_occ/epoch_4.pth \
  --onnx-file onnx/base_e2e_lidar_occ_trt_epoch4.onnx \
  --track-state-len 601 \
  --dump-input-dir dumped_inputs/base_e2e_lidar_occ_trt_trace_epoch4 \
  2>&1 | tee logs/export_e2e_lidar_occ_trt_epoch4.log
```

产物：

```text
UniAD/onnx/base_e2e_lidar_occ_trt_epoch4.onnx
UniAD/onnx/base_e2e_lidar_occ_trt_epoch4.repaired.onnx
UniAD/dumped_inputs/base_e2e_lidar_occ_trt_trace_epoch4/
```

PyTorch trace 输出的最后一个 tensor 是：

```text
seg_out: (1, 5, 1, 120, 160), int32
```

Dense 导出边界不包含 sparse encoder / backbone / neck，所以加载 checkpoint 时出现
frontend 相关的 `unexpected key` 警告属于预期。

### 6.3 TensorRT engine 构建

从仓库根目录执行：

```bash
TRT_PATH=/home/baojiali/Downloads/TensorRT-10.7.0.23
export LD_LIBRARY_PATH=$TRT_PATH/lib:${LD_LIBRARY_PATH-}

$TRT_PATH/bin/trtexec \
  --onnx=UniAD_train/UniAD/onnx/base_e2e_lidar_occ_backbone_neck_epoch4.onnx \
  --saveEngine=UniAD/engine/base_e2e_lidar_occ_backbone_neck_epoch4.engine \
  --fp16 \
  --skipInference \
  2>&1 | tee UniAD/logs/trtexec_base_e2e_lidar_occ_backbone_neck_epoch4.log

MIN=601
OPT=601
MAX=1201
TRACK_SHAPES="prev_track_intances0:Lx512,prev_track_intances1:Lx3,prev_track_intances3:L,prev_track_intances4:L,prev_track_intances5:L,prev_track_intances6:L,prev_track_intances8:L,prev_track_intances9:Lx10,prev_track_intances11:Lx4x256,prev_track_intances12:Lx4,prev_track_intances13:L"

$TRT_PATH/bin/trtexec \
  --onnx=UniAD/onnx/base_e2e_lidar_occ_trt_epoch4.repaired.onnx \
  --saveEngine=UniAD/engine/base_e2e_lidar_occ_trt_epoch4.engine \
  --staticPlugins=inference_app/enqueueV3/build/libuniad_plugin.so \
  --fp16 \
  --minShapes=${TRACK_SHAPES//L/${MIN}} \
  --optShapes=${TRACK_SHAPES//L/${OPT}} \
  --maxShapes=${TRACK_SHAPES//L/${MAX}} \
  --skipInference \
  2>&1 | tee UniAD/logs/trtexec_base_e2e_lidar_occ_trt_epoch4.log
```

已验证产物：

```text
UniAD/engine/base_e2e_lidar_occ_backbone_neck_epoch4.engine
UniAD/engine/base_e2e_lidar_occ_trt_epoch4.engine
```

### 6.4 10 帧全链路运行

从仓库根目录执行：

```bash
TRT_PATH=/home/baojiali/Downloads/TensorRT-10.7.0.23
export LD_LIBRARY_PATH=$TRT_PATH/lib:${LD_LIBRARY_PATH-}

OUT=UniAD_train/UniAD/output/base_e2e_lidar_occ_epoch4_10f_full_20260706

inference_app/sparse_lidar/build/uniad_lidar_e2e \
  UniAD_train/UniAD/onnx/base_e2e_lidar_occ_sparse_encoder_epoch4.onnx \
  UniAD/engine/base_e2e_lidar_occ_backbone_neck_epoch4.engine \
  UniAD/engine/base_e2e_lidar_occ_trt_epoch4.engine \
  inference_app/enqueueV3/build/libuniad_plugin.so \
  UniAD_train/UniAD/dumped_inputs/base_e2e_lidar_deploy_data_10f \
  "$OUT" 10 \
  --metadata-json UniAD_train/UniAD/dumped_inputs/base_e2e_lidar_deploy_data_10f \
  --gt-detections UniAD_train/UniAD/dumped_inputs/base_e2e_lidar_deploy_data_10f \
  --track-init-dir UniAD/dumped_inputs/base_e2e_lidar_occ_trt_trace_epoch4 \
  --track-state-len 601 \
  --max-track-state-len 1201 \
  --score-threshold 0.2 \
  --bev-max-draw 200 \
  2>&1 | tee UniAD/logs/uniad_lidar_e2e_occ_epoch4_10f_full_20260706.log
```

运行时检测到 E2E engine 包含 `seg_out`，会额外写出每帧：

```text
frame_000000_seg_out.bin
```

`seg_out.bin` 的 layout 是：

```text
int32[1, 5, 1, 120, 160]
```

本次验证结果：

```text
detections_txt:          10
bev_svg:                 10
bev_compare_svg:         10
traj_bin:                10
valid_traj_masks_bin:    10
seg_out_bin:             10

frame_000000_seg_out.bin sum=994, unique=[0, 1]
frame_000001_seg_out.bin sum=998, unique=[0, 1]
frame_000002_seg_out.bin sum=993, unique=[0, 1]
```

日志和输出目录：

```text
UniAD/logs/uniad_lidar_e2e_occ_epoch4_10f_full_20260706.log
UniAD_train/UniAD/output/base_e2e_lidar_occ_epoch4_10f_full_20260706/
```

### 6.5 OCC 预测预览

本次还把 10 帧 `seg_out.bin` 生成了 prediction-only 预览：

```text
UniAD_train/UniAD/output/base_e2e_lidar_occ_epoch4_10f_full_20260706/base_e2e_lidar_occ_seg_out_10f_epoch4_full_20260706.webm
```

为了确认 engine 输出的 OCC 坐标和场景目标位置一致，本次还生成了一版 overlay
预览：

```text
UniAD_train/UniAD/output/base_e2e_lidar_occ_epoch4_10f_full_20260706/base_e2e_lidar_occ_engine_overlay_10f_epoch4_full_20260706.webm
```

中间帧目录已经清理，只保留上述两个 OCC 专用最终 WebM。

overlay 约定：

```text
gray  = 当前帧 LiDAR points
green = GT detection boxes
red   = engine 输出的 seg_out
```

抽查 `frame_000000` 和 `frame_000005`，红色 OCC 主要落在 GT box 和点云目标附近，
未来 step 的位移连续，没有出现全空、全满或明显坐标翻转。

当前部署输入目录没有 occupancy GT，所以 C++ TRT runtime 暂时只能做这种定性可视化，
不能计算 occ IoU；需要 GT/Pred/TP/FP/FN 对比时，仍使用前面的
`visualize_e2e_occ.py` PyTorch 可视化脚本。

### 6.6 Detection / motion 统一固定画布可视化

OCC 的 `seg_out` 预览仍使用第 6.5 节的 OCC 专用脚本；detection、GT compare、
MotionHead trajectory 和 all-in-one 视图统一使用：

```text
inference_app/sparse_lidar/visualize_e2e_outputs.py
```

本工具固定输出 `1200x900`、2 fps，避免旧 SVG / Chrome 截图流程导致视频尺寸不一致
或出现白边：

```bash
python3 inference_app/sparse_lidar/visualize_e2e_outputs.py \
  UniAD_train/UniAD/output/base_e2e_lidar_occ_epoch4_10f_full_20260706 \
  --data-dir UniAD_train/UniAD/dumped_inputs/base_e2e_lidar_deploy_data_10f \
  --gt-dir UniAD_train/UniAD/dumped_inputs/base_e2e_lidar_deploy_data_10f \
  --num-frames 10 \
  --tag base_e2e_lidar_occ_10f_epoch4_full_20260706 \
  --modes pred compare motion motion_compare all \
  --width 1200 --height 900 --fps 2 --min-score 0.2
```

本机已生成：

```text
UniAD_train/UniAD/output/base_e2e_lidar_occ_epoch4_10f_full_20260706/base_e2e_lidar_occ_10f_epoch4_full_20260706_pred_fixed.webm
UniAD_train/UniAD/output/base_e2e_lidar_occ_epoch4_10f_full_20260706/base_e2e_lidar_occ_10f_epoch4_full_20260706_compare_fixed.webm
UniAD_train/UniAD/output/base_e2e_lidar_occ_epoch4_10f_full_20260706/base_e2e_lidar_occ_10f_epoch4_full_20260706_motion_fixed.webm
UniAD_train/UniAD/output/base_e2e_lidar_occ_epoch4_10f_full_20260706/base_e2e_lidar_occ_10f_epoch4_full_20260706_motion_compare_fixed.webm
UniAD_train/UniAD/output/base_e2e_lidar_occ_epoch4_10f_full_20260706/base_e2e_lidar_occ_10f_epoch4_full_20260706_all_fixed.webm
```
