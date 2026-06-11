/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: MIT
 */

#include <cuda_runtime.h>
#include <sys/stat.h>
#include <sys/types.h>

#include <cerrno>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <algorithm>
#include <iomanip>
#include <memory>
#include <sstream>
#include <string>
#include <vector>

#include "lidar_bev_visualizer.hpp"
#include "lidar_metadata.hpp"
#include "lidar_postprocess.hpp"
#include "lidar_runtime.hpp"
#include "tensorrt.hpp"

namespace {

struct CliArgs {
  std::string sparse_onnx_path;
  std::string frontend_engine_path;
  std::string dense_engine_path;
  std::string plugin_path;
  std::string raw_points_input;
  std::string output_dir;
  int num_frames = 0;
  std::vector<int> grid_size = {
      uniad_lidar::kDefaultGridZ,
      uniad_lidar::kDefaultGridY,
      uniad_lidar::kDefaultGridX};
  std::string shift_dir;
  std::string metadata_input;
  std::string gt_detections_input;
  uniad_lidar::DetectionDecodeConfig decode_config;
  uniad_lidar::BevVisualizationConfig bev_visualization_config;
};

void usage(const char* program) {
  std::fprintf(
      stderr,
      "Usage: %s <sparse.onnx> <frontend.engine> <dense.engine> <plugin.so> "
      "<raw_points_file_or_dir_or_pattern> <output_dir> <num_frames> "
      "[grid_z grid_y grid_x] [--shift-dir <dir>] "
      "[--metadata-json <file_or_dir_or_pattern>] "
      "[--gt-detections <file_or_dir_or_pattern>] "
      "[--score-threshold <score>] [--max-dets <count>] "
      "[--bev-max-draw <count>] [--bev-score-threshold <score>]\n",
      program);
  std::fprintf(
      stderr,
      "\nRaw points input may be a single file for num_frames=1, a directory, "
      "or a printf-style pattern such as raw_points_%%06d.bin.\n");
}

bool path_stat(const std::string& path, struct stat* sb) {
  return stat(path.c_str(), sb) == 0;
}

bool path_exists(const std::string& path) {
  struct stat sb;
  return path_stat(path, &sb);
}

bool is_directory(const std::string& path) {
  struct stat sb;
  return path_stat(path, &sb) && S_ISDIR(sb.st_mode);
}

bool is_regular_file(const std::string& path) {
  struct stat sb;
  return path_stat(path, &sb) && S_ISREG(sb.st_mode);
}

std::string zero_pad(int value) {
  std::ostringstream out;
  out << std::setw(6) << std::setfill('0') << value;
  return out.str();
}

void replace_all(std::string* text,
                 const std::string& from,
                 const std::string& to) {
  size_t pos = 0;
  while ((pos = text->find(from, pos)) != std::string::npos) {
    text->replace(pos, from.size(), to);
    pos += to.size();
  }
}

bool has_frame_token(const std::string& pattern) {
  return pattern.find("%") != std::string::npos ||
         pattern.find("{frame}") != std::string::npos ||
         pattern.find("{frame:06d}") != std::string::npos;
}

std::string format_frame_pattern(const std::string& pattern, int frame) {
  if (pattern.find("%") != std::string::npos) {
    char buffer[4096];
    const int written = std::snprintf(
        buffer, sizeof(buffer), pattern.c_str(), frame);
    uniad_lidar::require(written > 0 &&
                             static_cast<size_t>(written) < sizeof(buffer),
                         "Failed to format frame path pattern: " + pattern);
    return std::string(buffer);
  }

  std::string path = pattern;
  replace_all(&path, "{frame:06d}", zero_pad(frame));
  replace_all(&path, "{frame}", std::to_string(frame));
  return path;
}

std::string describe_candidates(const std::vector<std::string>& candidates) {
  std::string out;
  for (const std::string& candidate : candidates) {
    out += "\n  " + candidate;
  }
  return out;
}

std::string resolve_raw_points_path(
    const std::string& input, int frame, int num_frames) {
  if (has_frame_token(input)) {
    const std::string path = format_frame_pattern(input, frame);
    uniad_lidar::require(path_exists(path), "Missing raw points file: " + path);
    return path;
  }

  if (num_frames == 1 && is_regular_file(input)) {
    return input;
  }

  uniad_lidar::require(is_directory(input),
                       "Raw points input is not a readable directory: " + input);

  const std::string frame6 = zero_pad(frame);
  const std::vector<std::string> candidates = {
      uniad_lidar::join_path(input, "raw_points_" + std::to_string(frame) + ".bin"),
      uniad_lidar::join_path(input, "raw_points_" + frame6 + ".bin"),
      uniad_lidar::join_path(input, frame6 + ".bin"),
      uniad_lidar::join_path(uniad_lidar::join_path(input, frame6),
                             "raw_points_0.bin"),
      uniad_lidar::join_path(uniad_lidar::join_path(input, frame6),
                             "current/raw_points_0.bin"),
      uniad_lidar::join_path(uniad_lidar::join_path(input, "test_" + frame6),
                             "current/raw_points_0.bin"),
  };
  for (const std::string& candidate : candidates) {
    if (path_exists(candidate)) return candidate;
  }

  uniad_lidar::fail("Could not resolve raw points for frame " +
                    std::to_string(frame) + ". Tried:" +
                    describe_candidates(candidates));
}

void ensure_directory(const std::string& path) {
  if (path.empty() || is_directory(path)) return;

  std::string current;
  size_t start = 0;
  if (path[0] == '/') {
    current = "/";
    start = 1;
  }

  while (start <= path.size()) {
    const size_t end = path.find('/', start);
    const std::string part =
        path.substr(start, end == std::string::npos ? std::string::npos
                                                    : end - start);
    if (!part.empty()) {
      current = current.empty() ? part : uniad_lidar::join_path(current, part);
      if (!path_exists(current)) {
        if (mkdir(current.c_str(), 0775) != 0 && errno != EEXIST) {
          uniad_lidar::fail("Failed to create directory " + current + ": " +
                            std::strerror(errno));
        }
      }
      uniad_lidar::require(is_directory(current),
                           "Output path component is not a directory: " +
                               current);
    }
    if (end == std::string::npos) break;
    start = end + 1;
  }
}

std::string output_prefix(const std::string& output_dir, int frame) {
  return uniad_lidar::join_path(output_dir, "frame_" + zero_pad(frame));
}

std::vector<float> load_shift_or_zero(
    const std::string& shift_dir, int frame, size_t shift_numel) {
  if (shift_dir.empty()) {
    return std::vector<float>(shift_numel, 0.0f);
  }

  const std::string frame6 = zero_pad(frame);
  const std::vector<std::string> candidates = {
      uniad_lidar::join_path(shift_dir, "shift_" + std::to_string(frame) + ".bin"),
      uniad_lidar::join_path(shift_dir, "shift_" + frame6 + ".bin"),
      uniad_lidar::join_path(shift_dir, "frame_" + frame6 + "_shift.bin"),
      uniad_lidar::join_path(uniad_lidar::join_path(shift_dir, frame6),
                             "shift.bin"),
      uniad_lidar::join_path(uniad_lidar::join_path(shift_dir, frame6),
                             "dense/shift.bin"),
  };
  for (const std::string& candidate : candidates) {
    if (path_exists(candidate)) {
      return uniad_lidar::read_raw_float(candidate, shift_numel);
    }
  }
  uniad_lidar::fail("Could not resolve shift for frame " +
                    std::to_string(frame) + ". Tried:" +
                    describe_candidates(candidates));
}

std::string resolve_metadata_path(
    const std::string& input, int frame, int num_frames) {
  if (has_frame_token(input)) {
    const std::string path = format_frame_pattern(input, frame);
    uniad_lidar::require(path_exists(path), "Missing metadata JSON: " + path);
    return path;
  }

  if (num_frames == 1 && is_regular_file(input)) {
    return input;
  }

  uniad_lidar::require(is_directory(input),
                       "Metadata JSON input is not a readable directory: " +
                           input);

  const std::string frame6 = zero_pad(frame);
  std::vector<std::string> candidates;
  if (num_frames == 1) {
    candidates.push_back(uniad_lidar::join_path(input, "img_metas.json"));
  }
  candidates.push_back(
      uniad_lidar::join_path(
          uniad_lidar::join_path(input, "frame_" + frame6),
          "img_metas.json"));
  candidates.push_back(
      uniad_lidar::join_path(input, "frame_" + frame6 + ".json"));
  candidates.push_back(
      uniad_lidar::join_path(
          uniad_lidar::join_path(input, frame6),
          "img_metas.json"));
  candidates.push_back(uniad_lidar::join_path(input, frame6 + ".json"));
  candidates.push_back(
      uniad_lidar::join_path(input, "metadata_" + frame6 + ".json"));
  candidates.push_back(
      uniad_lidar::join_path(input, "img_metas_" + frame6 + ".json"));
  candidates.push_back(
      uniad_lidar::join_path(
          uniad_lidar::join_path(input, "test_" + frame6),
          "img_metas.json"));
  candidates.push_back(
      uniad_lidar::join_path(input, "test_" + frame6 + ".json"));

  for (const std::string& candidate : candidates) {
    if (path_exists(candidate)) return candidate;
  }
  uniad_lidar::fail("Could not resolve metadata JSON for frame " +
                    std::to_string(frame) + ". Tried:" +
                    describe_candidates(candidates));
}

std::string resolve_gt_detections_path(
    const std::string& input, int frame, int num_frames) {
  if (input.empty()) return "";
  if (has_frame_token(input)) {
    const std::string path = format_frame_pattern(input, frame);
    uniad_lidar::require(path_exists(path),
                         "Missing GT detections file: " + path);
    return path;
  }

  if (num_frames == 1 && is_regular_file(input)) {
    return input;
  }

  uniad_lidar::require(is_directory(input),
                       "GT detections input is not a readable directory: " +
                           input);
  const std::string frame6 = zero_pad(frame);
  const std::vector<std::string> candidates = {
      uniad_lidar::join_path(input, "gt_detections_" + std::to_string(frame) + ".txt"),
      uniad_lidar::join_path(input, "gt_detections_" + frame6 + ".txt"),
      uniad_lidar::join_path(input, "frame_" + frame6 + "_gt_detections.txt"),
      uniad_lidar::join_path(input, "frame_" + frame6 + "_gt.txt"),
      uniad_lidar::join_path(input, frame6 + ".txt"),
      uniad_lidar::join_path(uniad_lidar::join_path(input, frame6),
                             "gt_detections.txt"),
      uniad_lidar::join_path(uniad_lidar::join_path(input, "frame_" + frame6),
                             "gt_detections.txt"),
  };
  for (const std::string& candidate : candidates) {
    if (path_exists(candidate)) return candidate;
  }

  uniad_lidar::fail("Could not resolve GT detections for frame " +
                    std::to_string(frame) + ". Tried:" +
                    describe_candidates(candidates));
}

CliArgs parse_cli(int argc, char** argv) {
  if (argc < 8 || std::string(argv[1]) == "--help") {
    usage(argv[0]);
    std::exit(argc < 8 ? 2 : 0);
  }

  CliArgs args;
  args.sparse_onnx_path = argv[1];
  args.frontend_engine_path = argv[2];
  args.dense_engine_path = argv[3];
  args.plugin_path = argv[4];
  args.raw_points_input = argv[5];
  args.output_dir = argv[6];
  args.num_frames = std::atoi(argv[7]);
  uniad_lidar::require(args.num_frames > 0, "num_frames must be positive.");

  int cursor = 8;
  if (cursor + 2 < argc && std::string(argv[cursor]).find("--") != 0) {
    args.grid_size = {
        std::atoi(argv[cursor]),
        std::atoi(argv[cursor + 1]),
        std::atoi(argv[cursor + 2])};
    cursor += 3;
  }

  while (cursor < argc) {
    const std::string option = argv[cursor++];
    if (option == "--shift-dir") {
      uniad_lidar::require(cursor < argc, "--shift-dir requires a value.");
      args.shift_dir = argv[cursor++];
    } else if (option == "--metadata-json") {
      uniad_lidar::require(cursor < argc,
                           "--metadata-json requires a value.");
      args.metadata_input = argv[cursor++];
    } else if (option == "--gt-detections") {
      uniad_lidar::require(cursor < argc,
                           "--gt-detections requires a value.");
      args.gt_detections_input = argv[cursor++];
    } else if (option == "--score-threshold") {
      uniad_lidar::require(cursor < argc,
                           "--score-threshold requires a value.");
      args.decode_config.score_threshold = std::atof(argv[cursor++]);
    } else if (option == "--max-dets") {
      uniad_lidar::require(cursor < argc, "--max-dets requires a value.");
      args.decode_config.max_num = std::atoi(argv[cursor++]);
    } else if (option == "--bev-max-draw") {
      uniad_lidar::require(cursor < argc, "--bev-max-draw requires a value.");
      args.bev_visualization_config.max_draw = std::atoi(argv[cursor++]);
    } else if (option == "--bev-score-threshold") {
      uniad_lidar::require(cursor < argc,
                           "--bev-score-threshold requires a value.");
      args.bev_visualization_config.min_score = std::atof(argv[cursor++]);
    } else {
      usage(argv[0]);
      uniad_lidar::fail("Unknown option: " + option);
    }
  }
  return args;
}

}  // namespace

int main(int argc, char** argv) {
  const CliArgs args = parse_cli(argc, argv);
  ensure_directory(args.output_dir);

  cudaStream_t stream = nullptr;
  uniad_lidar::check_cuda(
      cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking),
      "cudaStreamCreateWithFlags");

  uniad_lidar::load_trt_plugins(args.plugin_path);
  uniad_lidar::SparseEncoder sparse_encoder(args.sparse_onnx_path, stream);

  std::shared_ptr<TensorRT::Engine> frontend_engine =
      TensorRT::load(args.frontend_engine_path);
  uniad_lidar::require(static_cast<bool>(frontend_engine),
                       "Failed to load LiDAR frontend TensorRT engine.");

  std::shared_ptr<TensorRT::Engine> dense_engine =
      TensorRT::load(args.dense_engine_path);
  uniad_lidar::require(static_cast<bool>(dense_engine),
                       "Failed to load dense TensorRT engine.");

  const size_t prev_numel =
      static_cast<size_t>(dense_engine->numel("prev_bev"));
  const size_t shift_numel =
      static_cast<size_t>(dense_engine->numel("shift"));
  const size_t use_prev_numel =
      static_cast<size_t>(dense_engine->numel("use_prev_bev"));
  std::vector<float> prev_bev(prev_numel, 0.0f);
  bool prev_bev_valid = false;
  const std::vector<std::string> output_names = {
      "bev_embed", "all_cls_scores", "all_bbox_preds"};

  if (!args.metadata_input.empty() && !args.shift_dir.empty()) {
    std::printf("[INFO] --metadata-json provided; metadata-derived shift "
                "takes precedence over --shift-dir.\n");
  } else if (args.shift_dir.empty() && args.metadata_input.empty()) {
    std::printf("[WARN] --shift-dir not provided; using zero shift for all frames.\n");
  }

  for (int frame = 0; frame < args.num_frames; ++frame) {
    const std::string raw_points_path = resolve_raw_points_path(
        args.raw_points_input, frame, args.num_frames);
    std::printf("[INFO] frame %d raw points: %s\n",
                frame, raw_points_path.c_str());

    uniad_lidar::SparseInputSpec sparse_input;
    sparse_input.use_raw_points = true;
    sparse_input.raw_points_path = raw_points_path;
    std::vector<float> middle_bev = sparse_encoder.forward(
        sparse_input, args.grid_size, stream);

    std::vector<float> lidar_bev = uniad_lidar::run_single_input_trt(
        frontend_engine,
        "LiDAR Backbone+Neck TensorRT",
        "middle_bev",
        "lidar_bev",
        middle_bev,
        stream,
        frame == 0);

    uniad_lidar::DenseBevInput dense_input;
    dense_input.lidar_bev = lidar_bev;
    dense_input.prev_bev = prev_bev;
    if (!args.metadata_input.empty()) {
      uniad_lidar::require(
          shift_numel == 2,
          "Metadata-derived shift expects dense engine shift input with 2 floats.");
      const std::string metadata_path = resolve_metadata_path(
          args.metadata_input, frame, args.num_frames);
      const uniad_lidar::FrameMetadata metadata =
          uniad_lidar::read_frame_metadata(metadata_path);
      if (metadata.has_prev_bev_exists && !metadata.prev_bev_exists) {
        std::fill(prev_bev.begin(), prev_bev.end(), 0.0f);
        prev_bev_valid = false;
        dense_input.prev_bev = prev_bev;
      }
      dense_input.shift = uniad_lidar::shift_from_metadata(metadata);
      uniad_lidar::require(
          dense_input.shift.size() == shift_numel,
          "Metadata-derived shift size does not match dense engine input.");
      const char* prev_label = metadata.has_prev_bev_exists
          ? (metadata.prev_bev_exists ? "true" : "false")
          : "missing";
      std::printf("[INFO] frame %d metadata: %s, prev_bev_exists=%s, "
                  "shift=(%.9g, %.9g), use_prev_bev=%d\n",
                  frame, metadata_path.c_str(), prev_label,
                  dense_input.shift[0], dense_input.shift[1],
                  prev_bev_valid ? 1 : 0);
    } else {
      dense_input.shift = load_shift_or_zero(
          args.shift_dir, frame, shift_numel);
    }
    dense_input.use_prev_bev.assign(use_prev_numel,
                                    prev_bev_valid ? 1.0f : 0.0f);

    uniad_lidar::TensorMap outputs = uniad_lidar::run_dense_bev_trt(
        dense_engine,
        dense_input,
        output_names,
        stream,
        "Dense-BEV TensorRT",
        frame == 0);

    const std::string prefix = output_prefix(args.output_dir, frame);
    uniad_lidar::write_raw_float(prefix + "_lidar_bev.bin", lidar_bev);
    for (const std::string& name : output_names) {
      uniad_lidar::write_raw_float(prefix + "_" + name + ".bin",
                                   outputs.at(name).data);
    }

    const std::vector<uniad_lidar::Detection> detections =
        uniad_lidar::decode_lidar_detections(
            outputs.at("all_cls_scores").data,
            outputs.at("all_cls_scores").shape,
            outputs.at("all_bbox_preds").data,
            outputs.at("all_bbox_preds").shape,
            args.decode_config);
    uniad_lidar::write_detections_txt(prefix + "_detections.txt",
                                      detections);
    uniad_lidar::write_bev_svg(prefix + "_bev.svg",
                               detections,
                               args.bev_visualization_config);
    if (!args.gt_detections_input.empty()) {
      const std::string gt_path = resolve_gt_detections_path(
          args.gt_detections_input, frame, args.num_frames);
      const std::vector<uniad_lidar::Detection> gt_detections =
          uniad_lidar::read_detections_txt(gt_path);
      uniad_lidar::write_bev_comparison_svg(
          prefix + "_bev_compare.svg",
          gt_detections,
          detections,
          args.bev_visualization_config);
    }

    prev_bev = outputs.at("bev_embed").data;
    prev_bev_valid = true;
    std::printf("[INFO] frame %d outputs written with prefix %s (%zu detections)\n",
                frame, prefix.c_str(), detections.size());
  }

  uniad_lidar::check_cuda(cudaStreamDestroy(stream), "cudaStreamDestroy");
  return 0;
}
