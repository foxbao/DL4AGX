/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: MIT
 */

#include "lidar_runtime.hpp"

#include <cuda_fp16.h>
#include <dlfcn.h>

#include <NvInferPlugin.h>

#include <algorithm>
#include <cmath>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <numeric>
#include <utility>

#include "gpu_voxelize.hpp"
#include "onnx-parser.hpp"
#include "spconv/engine.hpp"
#include "spconv/tensor.hpp"
#include "spconv/version.hpp"

namespace uniad_lidar {
namespace {

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

bool has_tensor(
    const std::shared_ptr<TensorRT::Engine>& engine,
    const std::string& name) {
  for (int i = 0; i < engine->num_bindings(); ++i) {
    if (engine->get_binding_name(i) == name) return true;
  }
  return false;
}

bool has_dynamic_dim(const std::vector<TRT_INT_TYPE>& dims) {
  return std::any_of(dims.begin(), dims.end(), [](TRT_INT_TYPE dim) {
    return dim < 0;
  });
}

bool is_track_state_name(const std::string& name) {
  return name.find("prev_track_intances") == 0;
}

bool is_detection_list_output(const std::string& name) {
  return name == "bboxes_dict_bboxes" || name == "scores" ||
         name == "labels" || name == "bbox_index" || name == "obj_idxes";
}

bool is_motion_list_output(const std::string& name) {
  return name == "traj_scores_0" || name == "traj_0" ||
         name == "traj_scores_1" || name == "traj_1" ||
         name == "traj_scores" || name == "traj" ||
         name == "valid_traj_masks";
}

bool is_planning_output(const std::string& name) {
  return name == "sdc_traj";
}

std::vector<TRT_INT_TYPE> output_allocation_dims(
    const std::string& name,
    const std::vector<TRT_INT_TYPE>& dims,
    int max_track_state_len,
    int max_detections) {
  require(max_track_state_len > 0, "max_track_state_len must be positive.");
  require(max_detections > 0, "max_detections must be positive.");
  std::vector<TRT_INT_TYPE> out = dims;
  for (size_t i = 0; i < out.size(); ++i) {
    if (i == 0 && is_track_state_name(name)) {
      out[i] = static_cast<TRT_INT_TYPE>(max_track_state_len);
    } else if (i == 0 && is_detection_list_output(name)) {
      out[i] = static_cast<TRT_INT_TYPE>(max_detections);
    } else if (i == 0 && is_motion_list_output(name)) {
      out[i] = static_cast<TRT_INT_TYPE>(max_detections);
    } else if (is_planning_output(name)) {
      const TRT_INT_TYPE fallback[] = {1, 6, 2};
      require(out.size() == 3,
              "Planning output sdc_traj must have shape 1x6x2.");
      out[i] = fallback[i];
    } else if (out[i] > 0) {
      continue;
    } else if (out[i] == 0) {
      continue;
    } else {
      fail("No allocation fallback for dynamic TensorRT output: " + name);
    }
  }
  return out;
}

std::string dtype_name(TensorRT::DType dtype) {
  switch (dtype) {
    case TensorRT::DType::FLOAT:
      return "float32";
    case TensorRT::DType::HALF:
      return "float16";
    case TensorRT::DType::INT32:
      return "int32";
    case TensorRT::DType::INT8:
      return "int8";
    case TensorRT::DType::BOOL:
      return "bool";
    case TensorRT::DType::UINT8:
      return "uint8";
    case TensorRT::DType::INT64:
      return "int64";
    default:
      return "unsupported";
  }
}

void require_tensor_numel(
    const NamedTensor& tensor,
    const std::string& name,
    TensorRT::DType expected_dtype) {
  const size_t expected_numel = product(tensor.shape);
  if (expected_dtype == TensorRT::DType::FLOAT) {
    require(tensor.data.size() == expected_numel,
            name + " float data size does not match shape.");
  } else if (expected_dtype == TensorRT::DType::INT32) {
    require(tensor.int_data.size() == expected_numel ||
                tensor.data.size() == expected_numel,
            name + " int32 data size does not match shape.");
  } else {
    fail(name + " uses unsupported TensorRT input dtype: " +
         dtype_name(expected_dtype));
  }
}

void copy_tensor_to_device(
    const NamedTensor& tensor,
    const std::string& name,
    TensorRT::DType expected_dtype,
    DeviceBuffer* device,
    cudaStream_t stream) {
  require_tensor_numel(tensor, name, expected_dtype);
  if (expected_dtype == TensorRT::DType::FLOAT) {
    if (!tensor.data.empty()) {
      copy_float_to_device(tensor.data, device, stream);
      return;
    }
    std::vector<float> converted(tensor.int_data.begin(), tensor.int_data.end());
    copy_float_to_device(converted, device, stream);
    return;
  }
  if (expected_dtype == TensorRT::DType::INT32) {
    if (!tensor.int_data.empty()) {
      copy_int32_to_device(tensor.int_data, device, stream);
      return;
    }
    std::vector<int32_t> converted(tensor.data.size());
    for (size_t i = 0; i < tensor.data.size(); ++i) {
      converted[i] = static_cast<int32_t>(std::lround(tensor.data[i]));
    }
    copy_int32_to_device(converted, device, stream);
    return;
  }
  fail(name + " uses unsupported TensorRT input dtype: " +
       dtype_name(expected_dtype));
}

}  // namespace

DeviceBuffer::DeviceBuffer(size_t size) { reset(size); }

DeviceBuffer::~DeviceBuffer() { reset(0); }

void DeviceBuffer::reset(size_t size) {
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

std::vector<float> read_raw_float(
    const std::string& path, size_t expected_numel) {
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

std::vector<int32_t> read_raw_int32(
    const std::string& path, size_t expected_numel) {
  std::ifstream in(path, std::ios::binary);
  require(static_cast<bool>(in), "Failed to open: " + path);
  in.seekg(0, std::ios::end);
  const size_t bytes = static_cast<size_t>(in.tellg());
  in.seekg(0, std::ios::beg);
  require(bytes == expected_numel * sizeof(int32_t),
          "Unexpected byte size for " + path + ": " + std::to_string(bytes));
  std::vector<int32_t> data(expected_numel);
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

void write_raw_int32(const std::string& path, const std::vector<int32_t>& data) {
  std::ofstream out(path, std::ios::binary);
  require(static_cast<bool>(out), "Failed to open output: " + path);
  out.write(reinterpret_cast<const char*>(data.data()),
            static_cast<std::streamsize>(data.size() * sizeof(int32_t)));
  require(static_cast<bool>(out), "Failed to write output: " + path);
}

NamedTensor read_raw_tensor(
    const std::string& path,
    TensorRT::DType dtype,
    const std::vector<TRT_INT_TYPE>& shape) {
  NamedTensor tensor;
  tensor.shape = shape;
  tensor.dtype = dtype;
  const size_t expected_numel = product(shape);
  if (dtype == TensorRT::DType::FLOAT) {
    tensor.data = read_raw_float(path, expected_numel);
    return tensor;
  }
  if (dtype == TensorRT::DType::INT32) {
    tensor.int_data = read_raw_int32(path, expected_numel);
    return tensor;
  }
  fail("Unsupported raw tensor dtype for " + path + ": " + dtype_name(dtype));
}

void write_raw_tensor(const std::string& path, const NamedTensor& tensor) {
  if (tensor.dtype == TensorRT::DType::INT32) {
    if (!tensor.int_data.empty()) {
      write_raw_int32(path, tensor.int_data);
      return;
    }
    std::vector<int32_t> converted(tensor.data.size());
    for (size_t i = 0; i < tensor.data.size(); ++i) {
      converted[i] = static_cast<int32_t>(std::lround(tensor.data[i]));
    }
    write_raw_int32(path, converted);
    return;
  }
  if (tensor.dtype == TensorRT::DType::FLOAT ||
      tensor.dtype == TensorRT::DType::HALF) {
    if (!tensor.data.empty()) {
      write_raw_float(path, tensor.data);
      return;
    }
    std::vector<float> converted(tensor.int_data.begin(), tensor.int_data.end());
    write_raw_float(path, converted);
    return;
  }
  fail("Unsupported raw tensor dtype for output: " + dtype_name(tensor.dtype));
}

RawVoxelInput voxelize_raw_points(const std::string& raw_points_path) {
  const std::vector<float> points = read_all_raw_float(raw_points_path);
  require(points.size() % kPointFeatureNum == 0,
          "Raw points file does not contain Nx4 float32 points: " +
              raw_points_path);

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

SparseEncoder::SparseEncoder(const std::string& onnx_path, cudaStream_t stream) {
  spconv::set_verbose(std::getenv("SPARSE_LIDAR_VERBOSE") != nullptr);
  // Build precision is a global switch for libspconv. Default FP16; set
  // SPARSE_LIDAR_INT8=1 to build INT8 (requires an ONNX whose SparseConvolution
  // nodes carry precision=int8 + weight/input dynamic ranges, produced by
  // tools/spconv_int8_calibrate.py). Kept as an env switch so the FP16 path and
  // all existing callers are unaffected by default.
  const bool use_int8 = std::getenv("SPARSE_LIDAR_INT8") != nullptr;
  engine_ = spconv::load_engine_from_onnx(
      onnx_path,
      use_int8 ? spconv::Precision::Int8 : spconv::Precision::Float16,
      stream, false);
  require(static_cast<bool>(engine_), "Failed to load sparse ONNX engine.");
  if (use_int8) {
    std::printf("[SPARSE] built libspconv engine with Precision::Int8\n");
  }

  if (std::getenv("SPARSE_LIDAR_GPU_VOXEL") != nullptr) {
    GpuVoxelizeConfig vc{};
    for (int i = 0; i < 6; ++i) vc.range[i] = kPointCloudRange[i];
    for (int i = 0; i < 3; ++i) vc.voxel_size[i] = kVoxelSize[i];
    vc.grid[0] = static_cast<int>(std::round(
        (kPointCloudRange[3] - kPointCloudRange[0]) / kVoxelSize[0]));
    vc.grid[1] = static_cast<int>(std::round(
        (kPointCloudRange[4] - kPointCloudRange[1]) / kVoxelSize[1]));
    vc.grid[2] = static_cast<int>(std::round(
        (kPointCloudRange[5] - kPointCloudRange[2]) / kVoxelSize[2]));
    vc.max_voxels = kMaxVoxels;
    vc.max_points_per_voxel = kMaxPointsPerVoxel;
    vc.feature_num = kPointFeatureNum;
    gpu_voxelizer_.reset(new GpuVoxelizer(vc));
    std::printf("[SPARSE] GPU voxelizer enabled (grid %dx%dx%d)\n",
                vc.grid[2], vc.grid[1], vc.grid[0]);
  }
}

SparseEncoder::~SparseEncoder() = default;

std::vector<float> SparseEncoder::forward(
    const SparseInputSpec& sparse_input,
    const std::vector<int>& grid_size,
    cudaStream_t stream) {
  spconv::Tensor features;
  spconv::Tensor indices;
  DeviceBuffer raw_features_device;
  DeviceBuffer raw_indices_device;
  std::vector<int64_t> feature_shape;
  std::vector<int64_t> index_shape;
  void* feature_ptr = nullptr;
  void* index_ptr = nullptr;

  // Split timing: voxelize (disk read + CPU hashing) vs spconv GPU forward.
  const auto t_begin = std::chrono::steady_clock::now();
  if (sparse_input.use_raw_points && gpu_voxelizer_) {
    // GPU voxelization path: read points to host, voxelize on GPU, feed the
    // device features/coors straight into spconv (no CPU hashing, no separate
    // H2D of voxel outputs).
    const std::vector<float> points =
        read_all_raw_float(sparse_input.raw_points_path);
    const size_t num_points = points.size() / kPointFeatureNum;
    void* gpu_features = nullptr;
    void* gpu_coors = nullptr;
    size_t gpu_num_voxels = 0;
    check_cuda(gpu_voxelizer_->voxelize(points.data(), num_points, stream,
                                        &gpu_features, &gpu_coors,
                                        &gpu_num_voxels),
               "GpuVoxelizer::voxelize");
    feature_shape = {static_cast<int64_t>(gpu_num_voxels), kPointFeatureNum};
    index_shape = {static_cast<int64_t>(gpu_num_voxels), 4};
    feature_ptr = gpu_features;
    index_ptr = gpu_coors;
    std::printf("raw points: %zu x %d from %s (GPU voxelize)\n",
                num_points, kPointFeatureNum,
                sparse_input.raw_points_path.c_str());
    std::printf("generated sparse input: %zu voxels, max_points_per_voxel=%d, "
                "max_voxels=%d\n",
                gpu_num_voxels, kMaxPointsPerVoxel, kMaxVoxels);
  } else if (sparse_input.use_raw_points) {
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
  const auto t_voxel_done = std::chrono::steady_clock::now();

  std::printf("libspconv version: %s\n", NVSPCONV_VERSION);
  std::printf("sparse features: %s dtype=%s\n",
              shape_string(feature_shape).c_str(),
              spconv::dtype_string(spconv::DataType::Float16));
  std::printf("sparse indices: %s dtype=%s\n",
              shape_string(index_shape).c_str(),
              spconv::dtype_string(spconv::DataType::Int32));
  std::printf("sparse grid_size: %d x %d x %d\n",
              grid_size[0], grid_size[1], grid_size[2]);

  engine_->input(0)->features().reference(
      feature_ptr, feature_shape, spconv::DataType::Float16);
  engine_->input(0)->indices().reference(
      index_ptr, index_shape, spconv::DataType::Int32);
  engine_->input(0)->set_grid_size(grid_size);
  const auto t_spconv_begin = std::chrono::steady_clock::now();
  engine_->forward(stream);
  check_cuda(cudaStreamSynchronize(stream),
             "cudaStreamSynchronize(sparse-forward)");
  const auto t_spconv_done = std::chrono::steady_clock::now();
  {
    using ms = std::chrono::duration<double, std::milli>;
    std::printf("[SPARSE SPLIT] voxelize+H2D: %.3f ms | spconv GPU forward: "
                "%.3f ms\n",
                ms(t_voxel_done - t_begin).count(),
                ms(t_spconv_done - t_spconv_begin).count());
  }

  spconv::Tensor& output_device = engine_->output(0)->features();
  std::printf("sparse dense output: %s dtype=%s\n",
              shape_string(output_device.shape).c_str(),
              spconv::dtype_string(output_device.dtype()));
  spconv::Tensor output_host = output_device.to_host(stream);
  check_cuda(cudaStreamSynchronize(stream),
             "cudaStreamSynchronize(sparse-to-host)");

  return tensor_to_float_vector(output_host);
}

std::vector<float> run_sparse_encoder(
    const std::string& onnx_path,
    const SparseInputSpec& sparse_input,
    const std::vector<int>& grid_size,
    cudaStream_t stream) {
  SparseEncoder encoder(onnx_path, stream);
  return encoder.forward(sparse_input, grid_size, stream);
}

void copy_float_to_device(
    const std::vector<float>& host,
    DeviceBuffer* device,
    cudaStream_t stream) {
  device->reset(host.size() * sizeof(float));
  if (host.empty()) return;
  check_cuda(cudaMemcpyAsync(device->ptr, host.data(), device->bytes,
                             cudaMemcpyHostToDevice, stream),
             "cudaMemcpyAsync(H2D)");
}

void copy_int32_to_device(
    const std::vector<int32_t>& host,
    DeviceBuffer* device,
    cudaStream_t stream) {
  device->reset(host.size() * sizeof(int32_t));
  if (host.empty()) return;
  check_cuda(cudaMemcpyAsync(device->ptr, host.data(), device->bytes,
                             cudaMemcpyHostToDevice, stream),
             "cudaMemcpyAsync(H2D-int32)");
}

std::vector<float> copy_device_to_float(
    const DeviceBuffer& device,
    size_t numel,
    TensorRT::DType dtype,
    cudaStream_t stream) {
  if (dtype == TensorRT::DType::FLOAT) {
    std::vector<float> host(numel);
    if (numel == 0) return host;
    check_cuda(cudaMemcpyAsync(host.data(), device.ptr, numel * sizeof(float),
                               cudaMemcpyDeviceToHost, stream),
               "cudaMemcpyAsync(D2H-float)");
    check_cuda(cudaStreamSynchronize(stream),
               "cudaStreamSynchronize(D2H-float)");
    return host;
  }
  if (dtype == TensorRT::DType::HALF) {
    std::vector<uint16_t> half_host(numel);
    if (numel == 0) return {};
    check_cuda(cudaMemcpyAsync(half_host.data(), device.ptr,
                               numel * sizeof(uint16_t),
                               cudaMemcpyDeviceToHost, stream),
               "cudaMemcpyAsync(D2H-half)");
    check_cuda(cudaStreamSynchronize(stream),
               "cudaStreamSynchronize(D2H-half)");
    std::vector<float> host(numel);
    for (size_t i = 0; i < numel; ++i) {
      host[i] = spconv::native_half2float(half_host[i]);
    }
    return host;
  }
  if (dtype == TensorRT::DType::INT32) {
    std::vector<int32_t> int_host(numel);
    if (numel == 0) return {};
    check_cuda(cudaMemcpyAsync(int_host.data(), device.ptr,
                               numel * sizeof(int32_t),
                               cudaMemcpyDeviceToHost, stream),
               "cudaMemcpyAsync(D2H-int32)");
    check_cuda(cudaStreamSynchronize(stream),
               "cudaStreamSynchronize(D2H-int32)");
    return std::vector<float>(int_host.begin(), int_host.end());
  }
  fail("Unsupported TensorRT output dtype for compare.");
}

NamedTensor copy_device_to_tensor(
    const DeviceBuffer& device,
    const std::vector<TRT_INT_TYPE>& shape,
    TensorRT::DType dtype,
    cudaStream_t stream) {
  NamedTensor tensor;
  tensor.shape = shape;
  tensor.dtype = dtype;
  const size_t numel = product(shape);
  if (dtype == TensorRT::DType::FLOAT ||
      dtype == TensorRT::DType::HALF) {
    tensor.data = copy_device_to_float(device, numel, dtype, stream);
    return tensor;
  }
  if (dtype == TensorRT::DType::INT32) {
    tensor.int_data.resize(numel);
    if (numel == 0) return tensor;
    check_cuda(cudaMemcpyAsync(tensor.int_data.data(), device.ptr,
                               numel * sizeof(int32_t),
                               cudaMemcpyDeviceToHost, stream),
               "cudaMemcpyAsync(D2H-tensor-int32)");
    check_cuda(cudaStreamSynchronize(stream),
               "cudaStreamSynchronize(D2H-tensor-int32)");
    return tensor;
  }
  fail("Unsupported TensorRT output dtype: " + dtype_name(dtype));
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
          name + " must be a float32 TensorRT input for this runtime.");
}

std::vector<float> run_single_input_trt(
    const std::shared_ptr<TensorRT::Engine>& engine,
    const std::string& label,
    const std::string& input_name,
    const std::string& output_name,
    const std::vector<float>& input,
    cudaStream_t stream,
    bool print_info) {
  if (print_info) {
    engine->print(label.c_str());
  }
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
  check_cuda(cudaStreamSynchronize(stream),
             "cudaStreamSynchronize(single-trt)");

  auto dims = output_dims;
  auto iter = dynamic_output_shapes.find(output_name);
  if (iter != dynamic_output_shapes.end()) {
    dims = iter->second;
  }
  if (print_info) {
    std::printf("%s output %s: %s\n", label.c_str(), output_name.c_str(),
                shape_string_trt(dims).c_str());
  }
  return copy_device_to_float(output_dev, product(dims), output_dtype, stream);
}

TensorMap run_dense_bev_trt(
    const std::shared_ptr<TensorRT::Engine>& engine,
    const DenseBevInput& input,
    const std::vector<std::string>& output_names,
    cudaStream_t stream,
    const std::string& label,
    bool print_info) {
  if (print_info) {
    engine->print(label.c_str());
  }

  require_float_input(engine, "lidar_bev");
  require_float_input(engine, "prev_bev");
  require_float_input(engine, "shift");
  require_float_input(engine, "use_prev_bev");

  require(input.lidar_bev.size() ==
              static_cast<size_t>(engine->numel("lidar_bev")),
          "lidar_bev numel does not match TensorRT input.");
  require(input.prev_bev.size() ==
              static_cast<size_t>(engine->numel("prev_bev")),
          "prev_bev numel does not match TensorRT input.");
  require(input.shift.size() ==
              static_cast<size_t>(engine->numel("shift")),
          "shift numel does not match TensorRT input.");
  require(input.use_prev_bev.size() ==
              static_cast<size_t>(engine->numel("use_prev_bev")),
          "use_prev_bev numel does not match TensorRT input.");

  DeviceBuffer lidar_dev;
  DeviceBuffer prev_dev;
  DeviceBuffer shift_dev;
  DeviceBuffer use_prev_dev;
  copy_float_to_device(input.lidar_bev, &lidar_dev, stream);
  copy_float_to_device(input.prev_bev, &prev_dev, stream);
  copy_float_to_device(input.shift, &shift_dev, stream);
  copy_float_to_device(input.use_prev_bev, &use_prev_dev, stream);

  std::unordered_map<std::string, const void*> bindings;
  bindings["lidar_bev"] = lidar_dev.ptr;
  bindings["prev_bev"] = prev_dev.ptr;
  bindings["shift"] = shift_dev.ptr;
  bindings["use_prev_bev"] = use_prev_dev.ptr;

  std::unordered_map<std::string, std::unique_ptr<DeviceBuffer>> output_buffers;
  for (const std::string& name : output_names) {
    const auto dims = engine->run_dims(name);
    const size_t numel = product(dims);
    const TensorRT::DType dtype = engine->dtype(name);
    output_buffers[name].reset(new DeviceBuffer(numel * dtype_size(dtype)));
    bindings[name] = output_buffers[name]->ptr;
    if (print_info) {
      std::printf("TRT output %s: %s\n",
                  name.c_str(), shape_string_trt(dims).c_str());
    }
  }

  std::unordered_map<std::string, std::vector<TRT_INT_TYPE>> dynamic_output_shapes;
  nv::EventTimer timer;
  require(engine->forward(bindings, dynamic_output_shapes, stream, false, timer),
          label + " enqueue failed.");
  check_cuda(cudaStreamSynchronize(stream), "cudaStreamSynchronize(trt-forward)");

  TensorMap outputs;
  for (const std::string& name : output_names) {
    auto dims = engine->run_dims(name);
    auto iter = dynamic_output_shapes.find(name);
    if (iter != dynamic_output_shapes.end()) {
      dims = iter->second;
    }
    const size_t numel = product(dims);
    const TensorRT::DType dtype = engine->dtype(name);
    NamedTensor tensor;
    tensor.shape = dims;
    tensor.dtype = dtype;
    tensor.data = copy_device_to_float(*output_buffers[name], numel, dtype, stream);
    outputs[name] = std::move(tensor);
  }
  return outputs;
}

TensorMap run_track_lidar_trt(
    const std::shared_ptr<TensorRT::Engine>& engine,
    const TrackLidarInput& input,
    const std::vector<std::string>& output_names,
    cudaStream_t stream,
    const std::string& label,
    bool print_info,
    int max_track_state_len,
    int max_detections) {
  if (print_info) {
    engine->print(label.c_str());
  }

  const std::vector<std::string> fixed_float_inputs = {
      "lidar_bev", "prev_bev", "shift", "use_prev_bev",
      "prev_timestamp", "prev_l2g_r_mat", "prev_l2g_t",
      "timestamp", "l2g_r_mat", "l2g_t"};
  for (const std::string& name : fixed_float_inputs) {
    require(has_tensor(engine, name),
            "Missing TensorRT input: " + name);
    require_float_input(engine, name);
  }
  require(has_tensor(engine, "max_obj_id"), "Missing TensorRT input: max_obj_id");
  require(engine->dtype("max_obj_id") == TensorRT::DType::INT32,
          "max_obj_id must be an int32 TensorRT input.");
  const bool has_command_input = has_tensor(engine, "command");
  if (has_command_input) {
    const TensorRT::DType dtype = engine->dtype("command");
    require(dtype == TensorRT::DType::FLOAT || dtype == TensorRT::DType::INT32,
            "command must be a float32 or int32 TensorRT input.");
    require(!input.command.empty(),
            "TensorRT engine expects command input, but runtime did not set it.");
  }

  const std::vector<std::string> state_inputs = {
      "prev_track_intances0", "prev_track_intances1", "prev_track_intances3",
      "prev_track_intances4", "prev_track_intances5", "prev_track_intances6",
      "prev_track_intances8", "prev_track_intances9", "prev_track_intances11",
      "prev_track_intances12", "prev_track_intances13"};
  for (const std::string& name : state_inputs) {
    require(has_tensor(engine, name), "Missing TensorRT input: " + name);
    const TensorRT::DType dtype = engine->dtype(name);
    require(dtype == TensorRT::DType::FLOAT || dtype == TensorRT::DType::INT32,
            name + " must be float32 or int32.");
    require(input.track_state.find(name) != input.track_state.end(),
            "Missing track state tensor: " + name);
    const std::vector<TRT_INT_TYPE> static_dims = engine->static_dims(name);
    if (has_dynamic_dim(static_dims)) {
      require(engine->set_run_dims(name, input.track_state.at(name).shape),
              "Failed to set dynamic shape for " + name);
    }
  }

  std::unordered_map<std::string, std::unique_ptr<DeviceBuffer>> input_buffers;
  std::unordered_map<std::string, const void*> bindings;

  auto add_fixed_float = [&](const std::string& name,
                             const std::vector<float>& values) {
    const size_t expected = static_cast<size_t>(engine->numel(name));
    require(values.size() == expected,
            name + " numel mismatch.");
    auto buffer = std::make_unique<DeviceBuffer>(values.size() * sizeof(float));
    copy_float_to_device(values, buffer.get(), stream);
    bindings[name] = buffer->ptr;
    input_buffers[name] = std::move(buffer);
  };
  auto add_fixed_int32 = [&](const std::string& name,
                             const std::vector<int32_t>& values) {
    const size_t expected = static_cast<size_t>(engine->numel(name));
    require(values.size() == expected,
            name + " numel mismatch.");
    auto buffer = std::make_unique<DeviceBuffer>(values.size() * sizeof(int32_t));
    copy_int32_to_device(values, buffer.get(), stream);
    bindings[name] = buffer->ptr;
    input_buffers[name] = std::move(buffer);
  };

  add_fixed_float("lidar_bev", input.lidar_bev);
  add_fixed_float("prev_bev", input.prev_bev);
  add_fixed_float("shift", input.shift);
  add_fixed_float("use_prev_bev", input.use_prev_bev);
  add_fixed_float("prev_timestamp", input.prev_timestamp);
  add_fixed_float("prev_l2g_r_mat", input.prev_l2g_r_mat);
  add_fixed_float("prev_l2g_t", input.prev_l2g_t);
  add_fixed_float("timestamp", input.timestamp);
  add_fixed_float("l2g_r_mat", input.l2g_r_mat);
  add_fixed_float("l2g_t", input.l2g_t);
  add_fixed_int32("max_obj_id", input.max_obj_id);
  if (has_command_input) {
    const TensorRT::DType dtype = engine->dtype("command");
    if (dtype == TensorRT::DType::FLOAT) {
      add_fixed_float("command", input.command);
    } else {
      std::vector<int32_t> command_int(input.command.size());
      for (size_t i = 0; i < input.command.size(); ++i) {
        command_int[i] = static_cast<int32_t>(std::lround(input.command[i]));
      }
      add_fixed_int32("command", command_int);
    }
  }

  for (const std::string& name : state_inputs) {
    const TensorRT::DType dtype = engine->dtype(name);
    const NamedTensor& tensor = input.track_state.at(name);
    require(tensor.shape == engine->run_dims(name),
            name + " runtime shape does not match requested input shape.");
    const size_t expected_numel = static_cast<size_t>(engine->numel(name));
    require(product(tensor.shape) == expected_numel,
            name + " dynamic numel mismatch.");
    auto buffer = std::make_unique<DeviceBuffer>(expected_numel * dtype_size(dtype));
    copy_tensor_to_device(tensor, name, dtype, buffer.get(), stream);
    bindings[name] = buffer->ptr;
    input_buffers[name] = std::move(buffer);
  }

  std::unordered_map<std::string, std::unique_ptr<DeviceBuffer>> output_buffers;
  for (const std::string& name : output_names) {
    require(has_tensor(engine, name), "Missing TensorRT output: " + name);
    const std::vector<TRT_INT_TYPE> run_shape = engine->run_dims(name);
    const std::vector<TRT_INT_TYPE> alloc_shape =
        output_allocation_dims(name, run_shape, max_track_state_len,
                               max_detections);
    const size_t alloc_numel = product(alloc_shape);
    const TensorRT::DType dtype = engine->dtype(name);
    output_buffers[name] =
        std::make_unique<DeviceBuffer>(alloc_numel * dtype_size(dtype));
    bindings[name] = output_buffers[name]->ptr;
    if (print_info) {
      std::printf("TRT output %s: run=%s alloc=%s dtype=%s\n",
                  name.c_str(), shape_string_trt(run_shape).c_str(),
                  shape_string_trt(alloc_shape).c_str(),
                  dtype_name(dtype).c_str());
    }
  }

  std::unordered_map<std::string, std::vector<TRT_INT_TYPE>> dynamic_output_shapes;
  nv::EventTimer timer;
  require(engine->forward(bindings, dynamic_output_shapes, stream, false, timer),
          label + " enqueue failed.");
  check_cuda(cudaStreamSynchronize(stream),
             "cudaStreamSynchronize(track-trt-forward)");

  TensorMap outputs;
  for (const std::string& name : output_names) {
    std::vector<TRT_INT_TYPE> dims = engine->run_dims(name);
    auto iter = dynamic_output_shapes.find(name);
    if (iter != dynamic_output_shapes.end()) {
      dims = iter->second;
    }
    require(!has_dynamic_dim(dims),
            "TensorRT did not resolve output shape for " + name);
    const size_t numel = product(dims);
    const TensorRT::DType dtype = engine->dtype(name);
    outputs[name] = copy_device_to_tensor(*output_buffers.at(name), dims,
                                          dtype, stream);
  }
  return outputs;
}

}  // namespace uniad_lidar
