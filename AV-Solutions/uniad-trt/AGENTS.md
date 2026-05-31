# Repository Guidelines

## Project Structure & Module Organization

This repository packages a UniAD-tiny deployment workflow for TensorRT. Root `documents/` holds setup, data, export, and quantization guides; `tools/` contains helper scripts for patching, metadata preparation, and ONNX export; `projects/configs/` contains UniAD-tiny training/export configs. C++/CUDA inference samples live in `inference_app/enqueueV2/` and `inference_app/enqueueV3/`, each with `src/`, `include/`, and `CMakeLists.txt`. `docker/` defines the Torch 1.12 environment, `patch/` stores integration patches, `onnx/` contains the dummy ONNX model, and `assets/` stores documentation media. `UniAD/` and `dependencies/BEVFormer_tensorrt/` have their own `AGENTS.md`; follow those files when editing inside those trees.

## Build, Test, and Development Commands

- `git submodule update --init --recursive`: fetch embedded UniAD, BEVFormer TensorRT, and nuScenes dependencies.
- `docker build -t uniad_torch1.12 -f docker/uniad_torch1.12.dockerfile docker`: build the documented Python/CUDA environment.
- `cd UniAD && PYTHONPATH=$(pwd) python3 ./tools/process_metadata.py --num_frame 69`: generate ONNX/TRT input dumps under `UniAD/nuscenes_np/`.
- `cd UniAD && CUDA_VISIBLE_DEVICES=0 ./tools/uniad_export_onnx.sh ./projects/configs/stage2_e2e/tiny_imgx0.25_e2e_trt_p.py ./ckpts/tiny_imgx0.25_e2e_ep20.pth 1`: export ONNX from a trained checkpoint.
- `cd inference_app/enqueueV3 && mkdir -p build && cd build && cmake .. -DTENSORRT_PATH=/path/to/TensorRT -DTARGET_GPU_SM=87 && make -j$(nproc)`: build the TensorRT plugins and inference binary. Use `enqueueV2` for TensorRT 8.6 and `enqueueV3` for TensorRT 10.x.

## Coding Style & Naming Conventions

Use 4-space indentation for Python and C++/CUDA, matching nearby files. Python functions, variables, and config keys use `snake_case`; C++ classes and structs use `PascalCase`. Keep deployment config names descriptive and lowercase, for example `tiny_imgx0.25_e2e_trt_p.py`. Preserve existing SPDX/NVIDIA license headers when modifying source files.

## Testing Guidelines

There is no broad unit-test suite. Validate Python changes by regenerating metadata or exporting ONNX with the smallest relevant frame/checkpoint set. Validate inference changes by rebuilding the selected `enqueueV*` app, creating an engine with `trtexec --skipInference`, then running `./build/uniad <engine> ./build/libuniad_plugin.so <input_path> <output_path> <frames>` on a short preprocessed sequence. Record TensorRT, CUDA, GPU SM, config, checkpoint, and frame count.

## Commit & Pull Request Guidelines

Recent history uses short imperative or descriptive subjects such as `Support IQ deprecation...`, `Add ... guide`, and `Update ... submodule`. Keep commits focused and avoid bundling generated data, engines, checkpoints, images, or logs. Pull requests should describe the affected workflow, list exact commands run, link related issues, and include output screenshots or videos when visualization behavior changes.

## Security & Configuration Tips

Do not commit nuScenes data, checkpoints, generated `*.bin`, `*.npy`, `*.engine`, or machine-specific TensorRT paths. Keep large artifacts in ignored runtime directories such as `data/`, `engines/`, `UniAD/ckpts/`, or external storage.
