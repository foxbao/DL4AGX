/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: MIT
 */

#include <cuda_runtime.h>
#include <dlfcn.h>

#include <NvInferPlugin.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <memory>
#include <numeric>
#include <string>
#include <unordered_map>
#include <vector>

#include "onnx-parser.hpp"
#include "spconv/engine.hpp"
#include "spconv/tensor.hpp"
#include "spconv/version.hpp"
#include "tensorrt.hpp"

namespace {

constexpr int kDefaultGridZ = 41;
constexpr int kDefaultGridY = 960;
constexpr int kDefaultGridX = 1280;

class TrtLogger : public nvinfer1::ILogger {
 public:
  void log(Severity severity, const char* msg) noexcept override {
    if (severity <= Severity::kWARNING) {
      std::fprintf(stderr, "[TRT] %s\n", msg);
    }
  }
};

void usage(const char* program) {
  std::fprintf(
      stderr,
      "Usage: %s <sparse.onnx> <features.tensor> <indices.tensor> "
      "<frontend.engine> <dense.engine> <plugin.so> <golden_run_dir> "
      "<output_prefix> [grid_z grid_y grid_x]\n",
      program);
}

[[noreturn]]
void fail(const std::string& message) {
  std::fprintf(stderr, "%s\n", message.c_str());
  std::exit(2);
}

void require(bool condition, const std::string& message) {
  if (!condition) fail(message);
}

void check_cuda(cudaError_t status, const char* call) {
  if (status != cudaSuccess) {
    std::fprintf(stderr, "CUDA error in %s: %s (%s)\n", call,
                 cudaGetErrorString(status), cudaGetErrorName(status));
    std::exit(2);
  }
}

std::string join_path(const std::string& dir, const std::string& name) {
  if (dir.empty() || dir.back() == '/') return dir + name;
  return dir + "/" + name;
}

std::string shape_string(const std::vector<int64_t>& shape) {
  std::string out;
  for (size_t i = 0; i < shape.size(); ++i) {
    if (i > 0) out += " x ";
    out += std::to_string(shape[i]);
  }
  return out;
}

std::string shape_string_trt(const std::vector<TRT_INT_TYPE>& shape) {
  std::string out;
  for (size_t i = 0; i < shape.size(); ++i) {
    if (i > 0) out += " x ";
    out += std::to_string(shape[i]);
  }
  return out;
}

size_t product(const std::vector<TRT_INT_TYPE>& shape) {
  return std::accumulate(shape.begin(), shape.end(), static_cast<size_t>(1),
                         [](size_t a, TRT_INT_TYPE b) {
                           return a * static_cast<size_t>(b);
                         });
}

size_t dtype_size(TensorRT::DType dtype) {
  switch (dtype) {
    case TensorRT::DType::FLOAT:
      return sizeof(float);
    case TensorRT::DType::HALF:
      return sizeof(uint16_t);
    case TensorRT::DType::INT32:
      return sizeof(int32_t);
    case TensorRT::DType::INT8:
    case TensorRT::DType::BOOL:
    case TensorRT::DType::UINT8:
      return 1;
    default:
      fail("Unsupported TensorRT dtype for buffer allocation.");
  }
}

std::vector<int> parse_grid(int argc, char** argv) {
  if (argc == 12) {
    return {std::atoi(argv[9]), std::atoi(argv[10]), std::atoi(argv[11])};
  }
  return {kDefaultGridZ, kDefaultGridY, kDefaultGridX};
}

std::vector<float> read_raw_float(const std::string& path, size_t expected_numel) {
  std::ifstream in(path, std::ios::binary);
  require(static_cast<bool>(in), "Failed to open: " + path);
  in.seekg(0, std::ios::end);
  const size_t bytes = static_cast<size_t>(in.tellg());
  in.seekg(0, std::ios::beg);
  require(bytes == expected_numel * sizeof(float),
          "Unexpected byte size for " + path + ": " + std::to_string(bytes));
  std::vector<float> data(expected_numel);
  in.read(reinterpret_cast<char*>(data.data()), bytes);
  require(static_cast<bool>(in), "Failed to read: " + path);
  return data;
}

void write_raw_float(const std::string& path, const std::vector<float>& data) {
  std::ofstream out(path, std::ios::binary);
  require(static_cast<bool>(out), "Failed to open output: " + path);
  out.write(reinterpret_cast<const char*>(data.data()),
            static_cast<std::streamsize>(data.size() * sizeof(float)));
  require(static_cast<bool>(out), "Failed to write output: " + path);
}

float tensor_value_as_float(const spconv::Tensor& tensor, size_t index) {
  if (tensor.dtype() == spconv::DataType::Float16) {
    return spconv::native_half2float(tensor.ptr<unsigned short>()[index]);
  }
  if (tensor.dtype() == spconv::DataType::Float32) {
    return tensor.ptr<float>()[index];
  }
  fail(std::string("Unsupported sparse tensor dtype: ") +
       spconv::dtype_string(tensor.dtype()));
}

std::vector<float> tensor_to_float_vector(const spconv::Tensor& tensor) {
  std::vector<float> output(tensor.numel);
  for (size_t i = 0; i < tensor.numel; ++i) {
    output[i] = tensor_value_as_float(tensor, i);
  }
  return output;
}

std::vector<float> run_sparse_encoder(
    const std::string& onnx_path,
    const std::string& features_path,
    const std::string& indices_path,
    const std::vector<int>& grid_size,
    cudaStream_t stream) {
  spconv::set_verbose(std::getenv("SPARSE_LIDAR_VERBOSE") != nullptr);
  std::shared_ptr<spconv::Engine> engine =
      spconv::load_engine_from_onnx(
          onnx_path, spconv::Precision::Float16, stream, false);
  require(static_cast<bool>(engine), "Failed to load sparse ONNX engine.");

  spconv::Tensor features = spconv::Tensor::load(features_path.c_str(), true, stream);
  spconv::Tensor indices = spconv::Tensor::load(indices_path.c_str(), true, stream);
  check_cuda(cudaStreamSynchronize(stream), "cudaStreamSynchronize(load-sparse)");

  std::printf("libspconv version: %s\n", NVSPCONV_VERSION);
  std::printf("sparse features: %s dtype=%s\n",
              shape_string(features.shape).c_str(),
              spconv::dtype_string(features.dtype()));
  std::printf("sparse indices: %s dtype=%s\n",
              shape_string(indices.shape).c_str(),
              spconv::dtype_string(indices.dtype()));
  std::printf("sparse grid_size: %d x %d x %d\n",
              grid_size[0], grid_size[1], grid_size[2]);

  engine->input(0)->features().reference(
      features.ptr(), features.shape, spconv::DataType::Float16);
  engine->input(0)->indices().reference(
      indices.ptr(), indices.shape, spconv::DataType::Int32);
  engine->input(0)->set_grid_size(grid_size);
  engine->forward(stream);
  check_cuda(cudaStreamSynchronize(stream), "cudaStreamSynchronize(sparse-forward)");

  spconv::Tensor& output_device = engine->output(0)->features();
  std::printf("sparse dense output: %s dtype=%s\n",
              shape_string(output_device.shape).c_str(),
              spconv::dtype_string(output_device.dtype()));
  spconv::Tensor output_host = output_device.to_host(stream);
  check_cuda(cudaStreamSynchronize(stream), "cudaStreamSynchronize(sparse-to-host)");

  return tensor_to_float_vector(output_host);
}

struct DeviceBuffer {
  void* ptr = nullptr;
  size_t bytes = 0;

  DeviceBuffer() = default;
  explicit DeviceBuffer(size_t size) { reset(size); }
  ~DeviceBuffer() { reset(0); }

  DeviceBuffer(const DeviceBuffer&) = delete;
  DeviceBuffer& operator=(const DeviceBuffer&) = delete;

  void reset(size_t size) {
    if (ptr != nullptr) {
      check_cuda(cudaFree(ptr), "cudaFree");
      ptr = nullptr;
      bytes = 0;
    }
    if (size > 0) {
      check_cuda(cudaMalloc(&ptr, size), "cudaMalloc");
      bytes = size;
    }
  }
};

void copy_float_to_device(
    const std::vector<float>& host,
    DeviceBuffer* device,
    cudaStream_t stream) {
  device->reset(host.size() * sizeof(float));
  check_cuda(cudaMemcpyAsync(device->ptr, host.data(), device->bytes,
                             cudaMemcpyHostToDevice, stream),
             "cudaMemcpyAsync(H2D)");
}

std::vector<float> copy_device_to_float(
    const DeviceBuffer& device,
    size_t numel,
    TensorRT::DType dtype,
    cudaStream_t stream) {
  if (dtype == TensorRT::DType::FLOAT) {
    std::vector<float> host(numel);
    check_cuda(cudaMemcpyAsync(host.data(), device.ptr, numel * sizeof(float),
                               cudaMemcpyDeviceToHost, stream),
               "cudaMemcpyAsync(D2H-float)");
    check_cuda(cudaStreamSynchronize(stream), "cudaStreamSynchronize(D2H-float)");
    return host;
  }
  if (dtype == TensorRT::DType::HALF) {
    std::vector<uint16_t> half_host(numel);
    check_cuda(cudaMemcpyAsync(half_host.data(), device.ptr,
                               numel * sizeof(uint16_t),
                               cudaMemcpyDeviceToHost, stream),
               "cudaMemcpyAsync(D2H-half)");
    check_cuda(cudaStreamSynchronize(stream), "cudaStreamSynchronize(D2H-half)");
    std::vector<float> host(numel);
    for (size_t i = 0; i < numel; ++i) {
      host[i] = spconv::native_half2float(half_host[i]);
    }
    return host;
  }
  fail("Unsupported TensorRT output dtype for compare.");
}

void compare_arrays(
    const std::string& name,
    const std::vector<float>& ref,
    const std::vector<float>& got) {
  require(ref.size() == got.size(), name + " output/reference size mismatch.");
  std::vector<float> abs_diffs(ref.size());
  std::vector<float> rel_diffs(ref.size());
  double abs_sum = 0.0;
  double rel_sum = 0.0;
  double ref_sq = 0.0;
  double got_sq = 0.0;
  double dot = 0.0;
  float max_abs = 0.0f;
  float max_rel = 0.0f;

  for (size_t i = 0; i < ref.size(); ++i) {
    const float abs_diff = std::abs(got[i] - ref[i]);
    const float rel_diff = abs_diff / std::max(std::abs(ref[i]), 1e-6f);
    abs_diffs[i] = abs_diff;
    rel_diffs[i] = rel_diff;
    abs_sum += abs_diff;
    rel_sum += rel_diff;
    max_abs = std::max(max_abs, abs_diff);
    max_rel = std::max(max_rel, rel_diff);
    ref_sq += static_cast<double>(ref[i]) * ref[i];
    got_sq += static_cast<double>(got[i]) * got[i];
    dot += static_cast<double>(got[i]) * ref[i];
  }

  std::sort(abs_diffs.begin(), abs_diffs.end());
  std::sort(rel_diffs.begin(), rel_diffs.end());
  const size_t p95_index = static_cast<size_t>(
      std::floor(0.95 * static_cast<double>(abs_diffs.size() - 1)));
  const size_t p99_index = static_cast<size_t>(
      std::floor(0.99 * static_cast<double>(abs_diffs.size() - 1)));
  const double cosine = dot / std::max(std::sqrt(ref_sq) * std::sqrt(got_sq), 1e-12);

  std::printf("%s:\n", name.c_str());
  std::printf("  max_abs: %.9f\n", max_abs);
  std::printf("  mean_abs: %.9f\n", abs_sum / std::max<size_t>(ref.size(), 1));
  std::printf("  p95_abs: %.9f\n", abs_diffs[p95_index]);
  std::printf("  p99_abs: %.9f\n", abs_diffs[p99_index]);
  std::printf("  max_rel: %.9f\n", max_rel);
  std::printf("  mean_rel: %.9f\n", rel_sum / std::max<size_t>(ref.size(), 1));
  std::printf("  p95_rel: %.9f\n", rel_diffs[p95_index]);
  std::printf("  cosine: %.9f\n", cosine);
}

void load_trt_plugins(const std::string& plugin_path) {
  static TrtLogger logger;
  require(initLibNvInferPlugins(&logger, ""),
          "Failed to initialize TensorRT standard plugins.");
  void* handle = dlopen(plugin_path.c_str(), RTLD_NOW | RTLD_GLOBAL);
  if (handle == nullptr) {
    fail(std::string("Failed to load TensorRT plugin library: ") + dlerror());
  }
  std::printf("loaded TensorRT plugin: %s\n", plugin_path.c_str());
}

void require_float_input(
    const std::shared_ptr<TensorRT::Engine>& engine,
    const std::string& name) {
  require(engine->dtype(name) == TensorRT::DType::FLOAT,
          name + " must be a float32 TensorRT input for this validator.");
}

std::vector<float> run_single_input_trt(
    const std::shared_ptr<TensorRT::Engine>& engine,
    const std::string& label,
    const std::string& input_name,
    const std::string& output_name,
    const std::vector<float>& input,
    cudaStream_t stream) {
  engine->print(label.c_str());
  require_float_input(engine, input_name);
  require(static_cast<size_t>(engine->numel(input_name)) == input.size(),
          label + " input numel mismatch.");

  DeviceBuffer input_dev;
  copy_float_to_device(input, &input_dev, stream);

  const auto output_dims = engine->run_dims(output_name);
  const size_t output_numel = product(output_dims);
  const TensorRT::DType output_dtype = engine->dtype(output_name);
  DeviceBuffer output_dev(output_numel * dtype_size(output_dtype));

  std::unordered_map<std::string, const void*> bindings;
  bindings[input_name] = input_dev.ptr;
  bindings[output_name] = output_dev.ptr;

  std::unordered_map<std::string, std::vector<TRT_INT_TYPE>> dynamic_output_shapes;
  nv::EventTimer timer;
  require(engine->forward(bindings, dynamic_output_shapes, stream, false, timer),
          label + " enqueue failed.");
  check_cuda(cudaStreamSynchronize(stream), "cudaStreamSynchronize(single-trt)");

  auto dims = output_dims;
  auto iter = dynamic_output_shapes.find(output_name);
  if (iter != dynamic_output_shapes.end()) {
    dims = iter->second;
  }
  std::printf("%s output %s: %s\n", label.c_str(), output_name.c_str(),
              shape_string_trt(dims).c_str());
  return copy_device_to_float(output_dev, product(dims), output_dtype, stream);
}

}  // namespace

int main(int argc, char** argv) {
  if (argc != 9 && argc != 12) {
    usage(argv[0]);
    return 2;
  }

  const std::string sparse_onnx_path = argv[1];
  const std::string features_path = argv[2];
  const std::string indices_path = argv[3];
  const std::string frontend_engine_path = argv[4];
  const std::string dense_engine_path = argv[5];
  const std::string plugin_path = argv[6];
  const std::string golden_run_dir = argv[7];
  const std::string output_prefix = argv[8];
  const std::vector<int> grid_size = parse_grid(argc, argv);

  cudaStream_t stream = nullptr;
  check_cuda(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking),
             "cudaStreamCreateWithFlags");

  std::vector<float> middle_bev = run_sparse_encoder(
      sparse_onnx_path, features_path, indices_path, grid_size, stream);

  load_trt_plugins(plugin_path);

  std::shared_ptr<TensorRT::Engine> frontend_engine =
      TensorRT::load(frontend_engine_path);
  require(static_cast<bool>(frontend_engine),
          "Failed to load LiDAR frontend TensorRT engine.");
  std::vector<float> lidar_bev = run_single_input_trt(
      frontend_engine,
      "LiDAR Backbone+Neck TensorRT",
      "middle_bev",
      "lidar_bev",
      middle_bev,
      stream);
  write_raw_float(output_prefix + "_lidar_bev.bin", lidar_bev);
  std::printf("saved %s\n", (output_prefix + "_lidar_bev.bin").c_str());
  std::printf("Compare LiDAR frontend output:\n");
  compare_arrays(
      "lidar_bev",
      read_raw_float(join_path(golden_run_dir, "current/lidar_bev.bin"),
                     lidar_bev.size()),
      lidar_bev);

  std::shared_ptr<TensorRT::Engine> dense_engine = TensorRT::load(dense_engine_path);
  require(static_cast<bool>(dense_engine), "Failed to load dense TensorRT engine.");
  dense_engine->print("Dense-BEV TensorRT");

  require_float_input(dense_engine, "lidar_bev");
  require_float_input(dense_engine, "prev_bev");
  require_float_input(dense_engine, "shift");
  require_float_input(dense_engine, "use_prev_bev");

  const size_t lidar_numel = static_cast<size_t>(dense_engine->numel("lidar_bev"));
  const size_t prev_numel = static_cast<size_t>(dense_engine->numel("prev_bev"));
  const size_t shift_numel = static_cast<size_t>(dense_engine->numel("shift"));
  const size_t use_prev_numel = static_cast<size_t>(dense_engine->numel("use_prev_bev"));
  require(lidar_bev.size() == lidar_numel,
          "Sparse dense BEV numel does not match TensorRT lidar_bev input.");

  const std::string dense_input_dir = join_path(golden_run_dir, "dense");
  std::vector<float> prev_bev = read_raw_float(
      join_path(dense_input_dir, "prev_bev.bin"), prev_numel);
  std::vector<float> shift = read_raw_float(
      join_path(dense_input_dir, "shift.bin"), shift_numel);
  std::vector<float> use_prev_bev = read_raw_float(
      join_path(dense_input_dir, "use_prev_bev.bin"), use_prev_numel);

  DeviceBuffer lidar_dev;
  DeviceBuffer prev_dev;
  DeviceBuffer shift_dev;
  DeviceBuffer use_prev_dev;
  copy_float_to_device(lidar_bev, &lidar_dev, stream);
  copy_float_to_device(prev_bev, &prev_dev, stream);
  copy_float_to_device(shift, &shift_dev, stream);
  copy_float_to_device(use_prev_bev, &use_prev_dev, stream);

  std::unordered_map<std::string, const void*> bindings;
  bindings["lidar_bev"] = lidar_dev.ptr;
  bindings["prev_bev"] = prev_dev.ptr;
  bindings["shift"] = shift_dev.ptr;
  bindings["use_prev_bev"] = use_prev_dev.ptr;

  const std::vector<std::string> output_names = {
      "bev_embed", "all_cls_scores", "all_bbox_preds"};
  std::unordered_map<std::string, std::unique_ptr<DeviceBuffer>> output_buffers;
  for (const std::string& name : output_names) {
    const auto dims = dense_engine->run_dims(name);
    const size_t numel = product(dims);
    const TensorRT::DType dtype = dense_engine->dtype(name);
    output_buffers[name] = std::unique_ptr<DeviceBuffer>(
        new DeviceBuffer(numel * dtype_size(dtype)));
    bindings[name] = output_buffers[name]->ptr;
    std::printf("TRT output %s: %s\n",
                name.c_str(), shape_string_trt(dims).c_str());
  }

  std::unordered_map<std::string, std::vector<TRT_INT_TYPE>> dynamic_output_shapes;
  nv::EventTimer timer;
  require(dense_engine->forward(bindings, dynamic_output_shapes, stream, false, timer),
          "Dense TensorRT enqueue failed.");
  check_cuda(cudaStreamSynchronize(stream), "cudaStreamSynchronize(trt-forward)");

  std::printf("Compare dense TensorRT outputs:\n");
  for (const std::string& name : output_names) {
    auto dims = dense_engine->run_dims(name);
    auto iter = dynamic_output_shapes.find(name);
    if (iter != dynamic_output_shapes.end()) {
      dims = iter->second;
    }
    const size_t numel = product(dims);
    const TensorRT::DType dtype = dense_engine->dtype(name);
    std::vector<float> got = copy_device_to_float(
        *output_buffers[name], numel, dtype, stream);
    write_raw_float(output_prefix + "_" + name + ".bin", got);
    std::printf("saved %s\n", (output_prefix + "_" + name + ".bin").c_str());
    std::vector<float> ref = read_raw_float(
        join_path(dense_input_dir, name + ".bin"), numel);
    compare_arrays(name, ref, got);
  }

  check_cuda(cudaStreamDestroy(stream), "cudaStreamDestroy");
  return 0;
}
