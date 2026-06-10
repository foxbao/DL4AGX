/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: MIT
 */

#include <cuda_runtime.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <memory>
#include <numeric>
#include <string>
#include <vector>

#include "onnx-parser.hpp"
#include "spconv/check.hpp"
#include "spconv/engine.hpp"
#include "spconv/tensor.hpp"
#include "spconv/version.hpp"

namespace {

void usage(const char* program) {
  std::fprintf(
      stderr,
      "Usage: %s <sparse.onnx> <features.tensor> <indices.tensor> "
      "<output.tensor> [reference.tensor] [grid_z grid_y grid_x]\n",
      program);
}

std::vector<int> parse_grid(int argc, char** argv) {
  if (argc >= 9) {
    return {std::atoi(argv[6]), std::atoi(argv[7]), std::atoi(argv[8])};
  }
  return {41, 960, 1280};
}

void check_cuda(cudaError_t status, const char* call) {
  if (status != cudaSuccess) {
    std::fprintf(stderr, "CUDA error in %s: %s (%s)\n", call,
                 cudaGetErrorString(status), cudaGetErrorName(status));
    std::exit(2);
  }
}

void require(bool condition, const char* message) {
  if (!condition) {
    std::fprintf(stderr, "%s\n", message);
    std::exit(2);
  }
}

std::string shape_string(const std::vector<int64_t>& shape) {
  std::string out;
  for (size_t i = 0; i < shape.size(); ++i) {
    if (i > 0) out += " x ";
    out += std::to_string(shape[i]);
  }
  return out;
}

float tensor_value_as_float(const spconv::Tensor& tensor, size_t index) {
  if (tensor.dtype() == spconv::DataType::Float16) {
    return spconv::native_half2float(tensor.ptr<unsigned short>()[index]);
  }
  if (tensor.dtype() == spconv::DataType::Float32) {
    return tensor.ptr<float>()[index];
  }
  std::fprintf(stderr, "Unsupported compare dtype: %s\n",
               spconv::dtype_string(tensor.dtype()));
  std::exit(2);
}

void compare_tensors(const spconv::Tensor& output, const spconv::Tensor& reference) {
  require(output.numel == reference.numel,
          "Output/reference numel mismatch.");
  require(output.shape == reference.shape,
          "Output/reference shape mismatch.");

  std::vector<float> diffs(output.numel);
  double sum = 0.0;
  double ref_sq = 0.0;
  double out_sq = 0.0;
  double dot = 0.0;
  float max_abs = 0.0f;

  for (size_t i = 0; i < output.numel; ++i) {
    const float out_value = tensor_value_as_float(output, i);
    const float ref_value = tensor_value_as_float(reference, i);
    const float diff = std::abs(out_value - ref_value);
    diffs[i] = diff;
    sum += diff;
    max_abs = std::max(max_abs, diff);
    ref_sq += static_cast<double>(ref_value) * ref_value;
    out_sq += static_cast<double>(out_value) * out_value;
    dot += static_cast<double>(out_value) * ref_value;
  }

  std::sort(diffs.begin(), diffs.end());
  const size_t p99_index = std::min(
      diffs.size() - 1,
      static_cast<size_t>(std::floor(0.99 * static_cast<double>(diffs.size() - 1))));
  const double cosine = dot / std::max(std::sqrt(ref_sq) * std::sqrt(out_sq), 1e-12);

  std::printf("Compare reference:\n");
  std::printf("  max_abs: %.9f\n", max_abs);
  std::printf("  mean_abs: %.9f\n", sum / std::max<size_t>(output.numel, 1));
  std::printf("  p99_abs: %.9f\n", diffs[p99_index]);
  std::printf("  cosine: %.9f\n", cosine);
}

}  // namespace

int main(int argc, char** argv) {
  if (argc != 5 && argc != 6 && argc != 9) {
    usage(argv[0]);
    return 2;
  }

  const std::string onnx_path = argv[1];
  const std::string features_path = argv[2];
  const std::string indices_path = argv[3];
  const std::string output_path = argv[4];
  const bool has_reference = argc >= 6;
  const std::string reference_path = has_reference ? argv[5] : "";
  const std::vector<int> grid_size = parse_grid(argc, argv);

  cudaStream_t stream = nullptr;
  check_cuda(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking),
             "cudaStreamCreateWithFlags");

  spconv::set_verbose(std::getenv("SPARSE_LIDAR_VERBOSE") != nullptr);
  std::shared_ptr<spconv::Engine> engine =
      spconv::load_engine_from_onnx(
          onnx_path, spconv::Precision::Float16, stream, false);
  require(static_cast<bool>(engine), "Failed to load sparse ONNX engine.");
  require(engine->num_input() == 1, "Expected one sparse engine input.");
  require(engine->num_output() >= 1, "Expected at least one sparse engine output.");

  spconv::Tensor features = spconv::Tensor::load(features_path.c_str(), true, stream);
  spconv::Tensor indices = spconv::Tensor::load(indices_path.c_str(), true, stream);
  check_cuda(cudaStreamSynchronize(stream), "cudaStreamSynchronize(load)");

  std::printf("libspconv version: %s\n", NVSPCONV_VERSION);
  std::printf("features: %s dtype=%s\n", shape_string(features.shape).c_str(),
              spconv::dtype_string(features.dtype()));
  std::printf("indices: %s dtype=%s\n", shape_string(indices.shape).c_str(),
              spconv::dtype_string(indices.dtype()));
  std::printf("grid_size: %d x %d x %d\n", grid_size[0], grid_size[1], grid_size[2]);

  engine->input(0)->features().reference(
      features.ptr(), features.shape, spconv::DataType::Float16);
  engine->input(0)->indices().reference(
      indices.ptr(), indices.shape, spconv::DataType::Int32);
  engine->input(0)->set_grid_size(grid_size);
  engine->forward(stream);
  check_cuda(cudaStreamSynchronize(stream), "cudaStreamSynchronize(forward)");

  spconv::Tensor& output_device = engine->output(0)->features();
  std::printf("output: %s dtype=%s\n", shape_string(output_device.shape).c_str(),
              spconv::dtype_string(output_device.dtype()));

  spconv::Tensor reference;
  if (has_reference) {
    reference = spconv::Tensor::load(reference_path.c_str(), false, stream);
    check_cuda(cudaStreamSynchronize(stream), "cudaStreamSynchronize(load-reference)");
    if (output_device.shape != reference.shape) {
      require(output_device.numel == reference.numel,
              "Output/reference shape mismatch and numel mismatch.");
      std::printf("reshape output metadata: %s -> %s\n",
                  shape_string(output_device.shape).c_str(),
                  shape_string(reference.shape).c_str());
      output_device.shape = reference.shape;
      output_device.ndim = reference.ndim;
    }
  }

  require(output_device.save(output_path.c_str(), stream),
          "Failed to save sparse output tensor.");
  check_cuda(cudaStreamSynchronize(stream), "cudaStreamSynchronize(save)");
  std::printf("saved output: %s\n", output_path.c_str());

  if (has_reference) {
    spconv::Tensor output_host = output_device.to_host(stream);
    check_cuda(cudaStreamSynchronize(stream), "cudaStreamSynchronize(compare)");
    compare_tensors(output_host, reference);
  }

  engine.reset();
  check_cuda(cudaStreamDestroy(stream), "cudaStreamDestroy");
  return 0;
}
