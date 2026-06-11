/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: MIT
 */

#include <cuda_runtime.h>

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <memory>
#include <string>
#include <vector>

#include "lidar_runtime.hpp"
#include "tensorrt.hpp"

namespace {

struct CliArgs {
  std::string sparse_onnx_path;
  uniad_lidar::SparseInputSpec sparse_input;
  std::string frontend_engine_path;
  std::string dense_engine_path;
  std::string plugin_path;
  std::string golden_run_dir;
  std::string output_prefix;
  std::vector<int> grid_size;
};

void usage(const char* program) {
  std::fprintf(
      stderr,
      "Usage: %s <sparse.onnx> (<features.tensor> <indices.tensor> | "
      "--raw-points <raw_points.bin>) <frontend.engine> <dense.engine> "
      "<plugin.so> <golden_run_dir> <output_prefix> [grid_z grid_y grid_x]\n",
      program);
}

std::vector<int> parse_grid(int argc, char** argv) {
  if (argc == 12) {
    return {std::atoi(argv[9]), std::atoi(argv[10]), std::atoi(argv[11])};
  }
  return {uniad_lidar::kDefaultGridZ,
          uniad_lidar::kDefaultGridY,
          uniad_lidar::kDefaultGridX};
}

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

void compare_arrays(
    const std::string& name,
    const std::vector<float>& ref,
    const std::vector<float>& got) {
  uniad_lidar::require(ref.size() == got.size(),
                       name + " output/reference size mismatch.");
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
  const double cosine = dot / std::max(std::sqrt(ref_sq) * std::sqrt(got_sq),
                                       1e-12);

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

}  // namespace

int main(int argc, char** argv) {
  const CliArgs args = parse_cli(argc, argv);

  cudaStream_t stream = nullptr;
  uniad_lidar::check_cuda(
      cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking),
      "cudaStreamCreateWithFlags");

  std::vector<float> middle_bev = uniad_lidar::run_sparse_encoder(
      args.sparse_onnx_path, args.sparse_input, args.grid_size, stream);

  uniad_lidar::load_trt_plugins(args.plugin_path);

  std::shared_ptr<TensorRT::Engine> frontend_engine =
      TensorRT::load(args.frontend_engine_path);
  uniad_lidar::require(static_cast<bool>(frontend_engine),
                       "Failed to load LiDAR frontend TensorRT engine.");
  std::vector<float> lidar_bev = uniad_lidar::run_single_input_trt(
      frontend_engine,
      "LiDAR Backbone+Neck TensorRT",
      "middle_bev",
      "lidar_bev",
      middle_bev,
      stream);
  uniad_lidar::write_raw_float(args.output_prefix + "_lidar_bev.bin",
                               lidar_bev);
  std::printf("saved %s\n", (args.output_prefix + "_lidar_bev.bin").c_str());
  std::printf("Compare LiDAR frontend output:\n");
  compare_arrays(
      "lidar_bev",
      uniad_lidar::read_raw_float(
          uniad_lidar::join_path(args.golden_run_dir, "current/lidar_bev.bin"),
          lidar_bev.size()),
      lidar_bev);

  std::shared_ptr<TensorRT::Engine> dense_engine =
      TensorRT::load(args.dense_engine_path);
  uniad_lidar::require(static_cast<bool>(dense_engine),
                       "Failed to load dense TensorRT engine.");

  const std::string dense_input_dir =
      uniad_lidar::join_path(args.golden_run_dir, "dense");
  uniad_lidar::DenseBevInput dense_input;
  dense_input.lidar_bev = std::move(lidar_bev);
  dense_input.prev_bev = uniad_lidar::read_raw_float(
      uniad_lidar::join_path(dense_input_dir, "prev_bev.bin"),
      static_cast<size_t>(dense_engine->numel("prev_bev")));
  dense_input.shift = uniad_lidar::read_raw_float(
      uniad_lidar::join_path(dense_input_dir, "shift.bin"),
      static_cast<size_t>(dense_engine->numel("shift")));
  dense_input.use_prev_bev = uniad_lidar::read_raw_float(
      uniad_lidar::join_path(dense_input_dir, "use_prev_bev.bin"),
      static_cast<size_t>(dense_engine->numel("use_prev_bev")));

  const std::vector<std::string> output_names = {
      "bev_embed", "all_cls_scores", "all_bbox_preds"};
  uniad_lidar::TensorMap outputs = uniad_lidar::run_dense_bev_trt(
      dense_engine, dense_input, output_names, stream);

  std::printf("Compare dense TensorRT outputs:\n");
  for (const std::string& name : output_names) {
    const auto& tensor = outputs.at(name);
    uniad_lidar::write_raw_float(args.output_prefix + "_" + name + ".bin",
                                 tensor.data);
    std::printf("saved %s\n",
                (args.output_prefix + "_" + name + ".bin").c_str());
    std::vector<float> ref = uniad_lidar::read_raw_float(
        uniad_lidar::join_path(dense_input_dir, name + ".bin"),
        tensor.data.size());
    compare_arrays(name, ref, tensor.data);
  }

  uniad_lidar::check_cuda(cudaStreamDestroy(stream), "cudaStreamDestroy");
  return 0;
}
