# UniAD-TensorRT 跑通流程

本文按从零到完成 TensorRT 推理的顺序整理本仓库的关键步骤。更详细背景可参考根目录 `README.md`、`documents/proj_setup.md`、`documents/env_prep.md`、`documents/data_prep.md`、`documents/train_export.md` 和 `inference_app/README.md`。

## 0. 准备条件

需要提前准备：

- NVIDIA GPU、CUDA、TensorRT 和 Docker。
- nuScenes 数据集。
- UniAD-tiny 训练权重。仓库中的 `onnx/uniad_tiny_dummy.onnx` 是随机权重，只适合验证流程。

TensorRT 8.6 使用 `inference_app/enqueueV2/`；TensorRT 10.x 使用 `inference_app/enqueueV3/`。

## 1. 初始化代码

```bash
git submodule update --init --recursive
```

## 2. 应用补丁和整理依赖

按 `documents/proj_setup.md` 执行。核心步骤如下：

```bash
cd UniAD
git apply --reject --whitespace=fix --exclude='*.DS_Store' ../patch/uniad-torch1.12.patch
git apply --reject --whitespace=fix --exclude='*.DS_Store' ../patch/uniad-onnx-export.patch
cd ..

cp -r ./dependencies/BEVFormer_tensorrt/third_party ./UniAD/
mv ./UniAD/third_party/bev_mmdet3d ./UniAD/third_party/uniad_mmdet3d

cd UniAD
git apply --exclude='*.DS_Store' ../patch/mmdet3d.patch
cd ..

chmod +x ./tools/add_bevformer_tensorrt_support.sh ./tools/add_onnx_export_support.sh
./tools/add_bevformer_tensorrt_support.sh
./tools/add_onnx_export_support.sh
chmod +x ./UniAD/tools/*.sh
```

## 3. 构建 Docker 环境

```bash
cd dependencies/nuscenes-devkit
git apply --exclude='*.DS_Store' ../../patch/nuscenes-devkit.patch
cp -r ./python-sdk/nuscenes ../../docker

cd ../../docker
docker build -t uniad_torch1.12 -f uniad_torch1.12.dockerfile .
```

## 4. 准备数据和预处理输入

将 nuScenes 放到：

```text
UniAD/data/nuscenes/
```

启动部署容器：

```bash
docker run -it --gpus all --shm-size=8g \
  -v /host/path/to/uniad-trt:/workspace/uniad-trt \
  uniad_torch1.12 /bin/bash
```

容器内编译 `uniad_mmdet3d`：

```bash
cd /workspace/uniad-trt/UniAD/third_party/uniad_mmdet3d/
python3 setup.py build develop --user
```

生成 ONNX 和 TensorRT 推理输入：

```bash
cd /workspace/uniad-trt/UniAD
PYTHONPATH=$(pwd) python3 ./tools/process_metadata.py --num_frame 69
```

输出目录：

```text
UniAD/nuscenes_np/uniad_onnx_input/
UniAD/nuscenes_np/uniad_trt_input/
```

## 5. 导出 ONNX

将真实训练权重放到：

```text
UniAD/ckpts/tiny_imgx0.25_e2e_ep20.pth
```

容器内执行：

```bash
cd /workspace/uniad-trt/UniAD
CUDA_VISIBLE_DEVICES=0 ./tools/uniad_export_onnx.sh \
  ./projects/configs/stage2_e2e/tiny_imgx0.25_e2e_trt_p.py \
  ./ckpts/tiny_imgx0.25_e2e_ep20.pth \
  1
```

## 6. 编译 TensorRT 插件和推理程序

以下示例使用 TensorRT 10.x 的 `enqueueV3`。Orin 常用 `TARGET_GPU_SM=87`，其他 GPU 请改成对应 compute capability。

```bash
cd inference_app/enqueueV3
mkdir -p build
cd build
cmake .. -DTENSORRT_PATH=/path/to/TensorRT -DTARGET_GPU_SM=87
make -j$(nproc)
```

生成文件包括：

```text
inference_app/enqueueV3/build/uniad
inference_app/enqueueV3/build/libuniad_plugin.so
```

## 7. 构建 TensorRT Engine

根据实际 ONNX 和 TensorRT 路径修改：

```bash
TRT_PATH=/path/to/TensorRT
ONNX_PATH=/path/to/uniad.onnx
ENGINE_PATH=/path/to/uniad.engine
PLUGIN_PATH=./build/libuniad_plugin.so

MIN=901
OPT=901
MAX=1150
SHAPES=prev_track_intances0:${MIN}x512,prev_track_intances1:${MIN}x3,prev_track_intances3:${MIN},prev_track_intances4:${MIN},prev_track_intances5:${MIN},prev_track_intances6:${MIN},prev_track_intances8:${MIN},prev_track_intances9:${MIN}x10,prev_track_intances11:${MIN}x4x256,prev_track_intances12:${MIN}x4,prev_track_intances13:${MIN}

LD_LIBRARY_PATH=${TRT_PATH}/lib:$LD_LIBRARY_PATH \
${TRT_PATH}/bin/trtexec \
  --onnx=${ONNX_PATH} \
  --saveEngine=${ENGINE_PATH} \
  --staticPlugins=${PLUGIN_PATH} \
  --verbose \
  --profilingVerbosity=detailed \
  --tacticSources=+CUBLAS \
  --minShapes=${SHAPES//${MIN}/${MIN}} \
  --optShapes=${SHAPES//${MIN}/${OPT}} \
  --maxShapes=${SHAPES//${MIN}/${MAX}} \
  --skipInference
```

## 8. 运行推理和查看结果

在 `inference_app/enqueueV3/` 目录下运行：

```bash
LD_LIBRARY_PATH=/path/to/TensorRT/lib/:$LD_LIBRARY_PATH \
./build/uniad \
  /path/to/uniad.engine \
  ./build/libuniad_plugin.so \
  /workspace/uniad-trt/UniAD/nuscenes_np/uniad_trt_input \
  /path/to/output \
  69
```

结果图片会写到：

```text
/path/to/output/dumped_video_results/
```

每张图片包含 6 路相机视角、检测框、BEV 视图和规划轨迹。

## 常见卡点

- 没有真实 checkpoint 时，只能验证流程，无法得到真实感知和规划效果。
- `TARGET_GPU_SM` 必须和实际 GPU 匹配。
- TensorRT 版本要和 `enqueueV2` / `enqueueV3` 选择一致。
- `uniad_trt_input/` 下必须有 `info.txt` 和各类 `.bin` 输入。
- `trtexec` 构建 engine 和 `./build/uniad` 运行时都要能找到 `libuniad_plugin.so`。
