/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: MIT
 */

#ifndef UNIAD_SPARSE_LIDAR_RUNTIME_HPP_
#define UNIAD_SPARSE_LIDAR_RUNTIME_HPP_

#include <cuda_runtime.h>

#include <cstddef>
#include <cstdint>
#include <memory>
#include <string>
#include <unordered_map>
#include <vector>

#include "tensorrt.hpp"

namespace spconv {
class Engine;
}  // namespace spconv

namespace uniad_lidar {

constexpr int kDefaultGridZ = 41;
constexpr int kDefaultGridY = 960;
constexpr int kDefaultGridX = 1280;
constexpr int kPointFeatureNum = 4;
constexpr int kMaxPointsPerVoxel = 10;
constexpr int kMaxVoxels = 160000;

struct DeviceBuffer {
  void* ptr = nullptr;
  size_t bytes = 0;

  DeviceBuffer() = default;
  explicit DeviceBuffer(size_t size);
  ~DeviceBuffer();

  DeviceBuffer(const DeviceBuffer&) = delete;
  DeviceBuffer& operator=(const DeviceBuffer&) = delete;

  void reset(size_t size);
};

struct SparseInputSpec {
  bool use_raw_points = false;
  std::string features_path;
  std::string indices_path;
  std::string raw_points_path;
};

struct RawVoxelInput {
  std::vector<uint16_t> features_half;
  std::vector<int32_t> coors;
  size_t num_points = 0;
  size_t num_voxels = 0;
};

struct DenseBevInput {
  std::vector<float> lidar_bev;
  std::vector<float> prev_bev;
  std::vector<float> shift;
  std::vector<float> use_prev_bev;
};

struct NamedTensor {
  std::vector<float> data;
  std::vector<TRT_INT_TYPE> shape;
  TensorRT::DType dtype = TensorRT::DType::NONE;
};

using TensorMap = std::unordered_map<std::string, NamedTensor>;

[[noreturn]]
void fail(const std::string& message);
void require(bool condition, const std::string& message);
void check_cuda(cudaError_t status, const char* call);

std::string join_path(const std::string& dir, const std::string& name);
std::string shape_string(const std::vector<int64_t>& shape);
std::string shape_string_trt(const std::vector<TRT_INT_TYPE>& shape);
size_t product(const std::vector<TRT_INT_TYPE>& shape);
size_t dtype_size(TensorRT::DType dtype);

std::vector<float> read_raw_float(
    const std::string& path, size_t expected_numel);
std::vector<float> read_all_raw_float(const std::string& path);
void write_raw_float(const std::string& path, const std::vector<float>& data);

RawVoxelInput voxelize_raw_points(const std::string& raw_points_path);

class SparseEncoder {
 public:
  SparseEncoder(const std::string& onnx_path, cudaStream_t stream);
  ~SparseEncoder();

  SparseEncoder(const SparseEncoder&) = delete;
  SparseEncoder& operator=(const SparseEncoder&) = delete;

  std::vector<float> forward(
      const SparseInputSpec& sparse_input,
      const std::vector<int>& grid_size,
      cudaStream_t stream);

 private:
  std::shared_ptr<spconv::Engine> engine_;
};

std::vector<float> run_sparse_encoder(
    const std::string& onnx_path,
    const SparseInputSpec& sparse_input,
    const std::vector<int>& grid_size,
    cudaStream_t stream);

void load_trt_plugins(const std::string& plugin_path);
void require_float_input(
    const std::shared_ptr<TensorRT::Engine>& engine,
    const std::string& name);

void copy_float_to_device(
    const std::vector<float>& host,
    DeviceBuffer* device,
    cudaStream_t stream);

std::vector<float> copy_device_to_float(
    const DeviceBuffer& device,
    size_t numel,
    TensorRT::DType dtype,
    cudaStream_t stream);

std::vector<float> run_single_input_trt(
    const std::shared_ptr<TensorRT::Engine>& engine,
    const std::string& label,
    const std::string& input_name,
    const std::string& output_name,
    const std::vector<float>& input,
    cudaStream_t stream,
    bool print_info = true);

TensorMap run_dense_bev_trt(
    const std::shared_ptr<TensorRT::Engine>& engine,
    const DenseBevInput& input,
    const std::vector<std::string>& output_names,
    cudaStream_t stream,
    const std::string& label = "Dense-BEV TensorRT",
    bool print_info = true);

}  // namespace uniad_lidar

#endif  // UNIAD_SPARSE_LIDAR_RUNTIME_HPP_
