/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: MIT
 */

#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <dlfcn.h>

#include <NvInferPlugin.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
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
constexpr int kPointFeatureNum = 4;
constexpr int kMaxPointsPerVoxel = 10;
constexpr int kMaxVoxels = 160000;
constexpr float kPointCloudRange[6] = {-64.0f, -48.0f, -2.0f,
                                       64.0f, 48.0f, 6.0f};
constexpr float kVoxelSize[3] = {0.1f, 0.1f, 0.2f};

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
      "Usage: %s <sparse.onnx> (<features.tensor> <indices.tensor> | "
      "--raw-points <raw_points.bin>) <frontend.engine> <dense.engine> "
      "<plugin.so> <golden_run_dir> <output_prefix> [grid_z grid_y grid_x]\n",
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

std::vector<float> read_all_raw_float(const std::string& path) {
  std::ifstream in(path, std::ios::binary);
  require(static_cast<bool>(in), "Failed to open: " + path);
  in.seekg(0, std::ios::end);
  const size_t bytes = static_cast<size_t>(in.tellg());
  in.seekg(0, std::ios::beg);
  require(bytes % sizeof(float) == 0,
          "Raw float file byte size is not divisible by sizeof(float): " + path);
  std::vector<float> data(bytes / sizeof(float));
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

uint16_t float_to_half_bits(float value) {
  const __half half_value = __float2half_rn(value);
  uint16_t bits = 0;
  std::memcpy(&bits, &half_value, sizeof(bits));
  return bits;
}

struct SparseInputSpec {
  bool use_raw_points = false;
  std::string features_path;
  std::string indices_path;
  std::string raw_points_path;
};

struct CliArgs {
  std::string sparse_onnx_path;
  SparseInputSpec sparse_input;
  std::string frontend_engine_path;
  std::string dense_engine_path;
  std::string plugin_path;
  std::string golden_run_dir;
  std::string output_prefix;
  std::vector<int> grid_size;
};

struct RawVoxelInput {
  std::vector<uint16_t> features_half;
  std::vector<int32_t> coors;
  size_t num_points = 0;
  size_t num_voxels = 0;
};

CliArgs parse_cli(int argc, char** argv) {
  if (argc != 9 && argc != 12) {
    usage(argv[0]);
    std::exit(2);
  }

  CliArgs args;
  args.sparse_onnx_path = argv[1];
  args.grid_size = parse_grid(argc, argv);
  if (std::string(argv[2]) == "--raw-points") {
    args.sparse_input.use_raw_points = true;
    args.sparse_input.raw_points_path = argv[3];
    args.frontend_engine_path = argv[4];
    args.dense_engine_path = argv[5];
    args.plugin_path = argv[6];
    args.golden_run_dir = argv[7];
    args.output_prefix = argv[8];
  } else {
    args.sparse_input.features_path = argv[2];
    args.sparse_input.indices_path = argv[3];
    args.frontend_engine_path = argv[4];
    args.dense_engine_path = argv[5];
    args.plugin_path = argv[6];
    args.golden_run_dir = argv[7];
    args.output_prefix = argv[8];
  }
  return args;
}

RawVoxelInput voxelize_raw_points(const std::string& raw_points_path) {
  const std::vector<float> points = read_all_raw_float(raw_points_path);
  require(points.size() % kPointFeatureNum == 0,
          "Raw points file does not contain Nx4 float32 points: " + raw_points_path);

  const size_t num_points = points.size() / kPointFeatureNum;
  std::unordered_map<int64_t, int32_t> voxel_index_by_key;
  voxel_index_by_key.reserve(std::min(num_points, static_cast<size_t>(kMaxVoxels)));
  std::vector<float> feature_sums;
  std::vector<int32_t> point_counts;
  std::vector<int32_t> coors;
  feature_sums.reserve(static_cast<size_t>(kMaxVoxels) * kPointFeatureNum);
  point_counts.reserve(kMaxVoxels);
  coors.reserve(static_cast<size_t>(kMaxVoxels) * 4);

  const int grid_x = static_cast<int>(
      std::round((kPointCloudRange[3] - kPointCloudRange[0]) / kVoxelSize[0]));
  const int grid_y = static_cast<int>(
      std::round((kPointCloudRange[4] - kPointCloudRange[1]) / kVoxelSize[1]));
  const int grid_z = static_cast<int>(
      std::round((kPointCloudRange[5] - kPointCloudRange[2]) / kVoxelSize[2]));

  for (size_t point_idx = 0; point_idx < num_points; ++point_idx) {
    const float* point = points.data() + point_idx * kPointFeatureNum;
    const float px = point[0];
    const float py = point[1];
    const float pz = point[2];
    if (px < kPointCloudRange[0] || px >= kPointCloudRange[3] ||
        py < kPointCloudRange[1] || py >= kPointCloudRange[4] ||
        pz < kPointCloudRange[2] || pz >= kPointCloudRange[5]) {
      continue;
    }

    const int cx = static_cast<int>(
        std::floor((px - kPointCloudRange[0]) / kVoxelSize[0]));
    const int cy = static_cast<int>(
        std::floor((py - kPointCloudRange[1]) / kVoxelSize[1]));
    const int cz = static_cast<int>(
        std::floor((pz - kPointCloudRange[2]) / kVoxelSize[2]));
    if (cx < 0 || cx >= grid_x || cy < 0 || cy >= grid_y ||
        cz < 0 || cz >= grid_z) {
      continue;
    }

    const int64_t voxel_key =
        (static_cast<int64_t>(cz) * grid_y + cy) * grid_x + cx;
    auto iter = voxel_index_by_key.find(voxel_key);
    int32_t voxel_index = 0;
    if (iter == voxel_index_by_key.end()) {
      if (point_counts.size() >= static_cast<size_t>(kMaxVoxels)) {
        continue;
      }
      voxel_index = static_cast<int32_t>(point_counts.size());
      voxel_index_by_key.emplace(voxel_key, voxel_index);
      coors.push_back(0);
      coors.push_back(cz);
      coors.push_back(cy);
      coors.push_back(cx);
      point_counts.push_back(0);
      for (int feature = 0; feature < kPointFeatureNum; ++feature) {
        feature_sums.push_back(0.0f);
      }
    } else {
      voxel_index = iter->second;
    }

    int32_t& point_count = point_counts[voxel_index];
    if (point_count >= kMaxPointsPerVoxel) {
      continue;
    }
    const size_t feature_base =
        static_cast<size_t>(voxel_index) * kPointFeatureNum;
    for (int feature = 0; feature < kPointFeatureNum; ++feature) {
      feature_sums[feature_base + feature] += point[feature];
    }
    ++point_count;
  }

  const size_t num_voxels = point_counts.size();
  require(num_voxels > 0, "Raw point voxelization produced no voxels.");

  RawVoxelInput output;
  output.num_points = num_points;
  output.num_voxels = num_voxels;
  output.coors = std::move(coors);
  output.features_half.resize(num_voxels * kPointFeatureNum);
  for (size_t voxel = 0; voxel < num_voxels; ++voxel) {
    const int32_t point_count = point_counts[voxel];
    require(point_count > 0, "Internal error: empty generated voxel.");
    const size_t feature_base = voxel * kPointFeatureNum;
    for (int feature = 0; feature < kPointFeatureNum; ++feature) {
      const float mean = feature_sums[feature_base + feature] /
                         static_cast<float>(point_count);
      output.features_half[feature_base + feature] = float_to_half_bits(mean);
    }
  }
  return output;
}

std::vector<float> run_sparse_encoder(
    const std::string& onnx_path,
    const SparseInputSpec& sparse_input,
    const std::vector<int>& grid_size,
    cudaStream_t stream) {
  spconv::set_verbose(std::getenv("SPARSE_LIDAR_VERBOSE") != nullptr);
  std::shared_ptr<spconv::Engine> engine =
      spconv::load_engine_from_onnx(
          onnx_path, spconv::Precision::Float16, stream, false);
  require(static_cast<bool>(engine), "Failed to load sparse ONNX engine.");

  spconv::Tensor features;
  spconv::Tensor indices;
  DeviceBuffer raw_features_device;
  DeviceBuffer raw_indices_device;
  std::vector<int64_t> feature_shape;
  std::vector<int64_t> index_shape;
  void* feature_ptr = nullptr;
  void* index_ptr = nullptr;

  if (sparse_input.use_raw_points) {
    RawVoxelInput raw_input = voxelize_raw_points(sparse_input.raw_points_path);
    raw_features_device.reset(raw_input.features_half.size() * sizeof(uint16_t));
    raw_indices_device.reset(raw_input.coors.size() * sizeof(int32_t));
    check_cuda(cudaMemcpyAsync(raw_features_device.ptr,
                               raw_input.features_half.data(),
                               raw_features_device.bytes,
                               cudaMemcpyHostToDevice, stream),
               "cudaMemcpyAsync(raw-features-H2D)");
    check_cuda(cudaMemcpyAsync(raw_indices_device.ptr,
                               raw_input.coors.data(),
                               raw_indices_device.bytes,
                               cudaMemcpyHostToDevice, stream),
               "cudaMemcpyAsync(raw-coors-H2D)");
    feature_shape = {static_cast<int64_t>(raw_input.num_voxels),
                     kPointFeatureNum};
    index_shape = {static_cast<int64_t>(raw_input.num_voxels), 4};
    feature_ptr = raw_features_device.ptr;
    index_ptr = raw_indices_device.ptr;
    std::printf("raw points: %zu x %d from %s\n",
                raw_input.num_points, kPointFeatureNum,
                sparse_input.raw_points_path.c_str());
    std::printf("generated sparse input: %zu voxels, max_points_per_voxel=%d, "
                "max_voxels=%d\n",
                raw_input.num_voxels, kMaxPointsPerVoxel, kMaxVoxels);
  } else {
    features = spconv::Tensor::load(
        sparse_input.features_path.c_str(), true, stream);
    indices = spconv::Tensor::load(
        sparse_input.indices_path.c_str(), true, stream);
    feature_shape = features.shape;
    index_shape = indices.shape;
    feature_ptr = features.ptr();
    index_ptr = indices.ptr();
  }
  check_cuda(cudaStreamSynchronize(stream), "cudaStreamSynchronize(load-sparse)");

  std::printf("libspconv version: %s\n", NVSPCONV_VERSION);
  std::printf("sparse features: %s dtype=%s\n",
              shape_string(feature_shape).c_str(),
              spconv::dtype_string(spconv::DataType::Float16));
  std::printf("sparse indices: %s dtype=%s\n",
              shape_string(index_shape).c_str(),
              spconv::dtype_string(spconv::DataType::Int32));
  std::printf("sparse grid_size: %d x %d x %d\n",
              grid_size[0], grid_size[1], grid_size[2]);

  engine->input(0)->features().reference(
      feature_ptr, feature_shape, spconv::DataType::Float16);
  engine->input(0)->indices().reference(
      index_ptr, index_shape, spconv::DataType::Int32);
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
  const CliArgs args = parse_cli(argc, argv);

  cudaStream_t stream = nullptr;
  check_cuda(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking),
             "cudaStreamCreateWithFlags");

  std::vector<float> middle_bev = run_sparse_encoder(
      args.sparse_onnx_path, args.sparse_input, args.grid_size, stream);

  load_trt_plugins(args.plugin_path);

  std::shared_ptr<TensorRT::Engine> frontend_engine =
      TensorRT::load(args.frontend_engine_path);
  require(static_cast<bool>(frontend_engine),
          "Failed to load LiDAR frontend TensorRT engine.");
  std::vector<float> lidar_bev = run_single_input_trt(
      frontend_engine,
      "LiDAR Backbone+Neck TensorRT",
      "middle_bev",
      "lidar_bev",
      middle_bev,
      stream);
  write_raw_float(args.output_prefix + "_lidar_bev.bin", lidar_bev);
  std::printf("saved %s\n", (args.output_prefix + "_lidar_bev.bin").c_str());
  std::printf("Compare LiDAR frontend output:\n");
  compare_arrays(
      "lidar_bev",
      read_raw_float(join_path(args.golden_run_dir, "current/lidar_bev.bin"),
                     lidar_bev.size()),
      lidar_bev);

  std::shared_ptr<TensorRT::Engine> dense_engine =
      TensorRT::load(args.dense_engine_path);
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

  const std::string dense_input_dir = join_path(args.golden_run_dir, "dense");
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
    write_raw_float(args.output_prefix + "_" + name + ".bin", got);
    std::printf("saved %s\n", (args.output_prefix + "_" + name + ".bin").c_str());
    std::vector<float> ref = read_raw_float(
        join_path(dense_input_dir, name + ".bin"), numel);
    compare_arrays(name, ref, got);
  }

  check_cuda(cudaStreamDestroy(stream), "cudaStreamDestroy");
  return 0;
}
