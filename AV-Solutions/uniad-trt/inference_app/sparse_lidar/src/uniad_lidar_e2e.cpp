/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: MIT
 */

#include <cuda_runtime.h>
#include <sys/stat.h>
#include <sys/types.h>

#include <algorithm>
#include <cerrno>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <iomanip>
#include <limits>
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

const std::vector<std::string>& track_state_input_names() {
  static const std::vector<std::string> names = {
      "prev_track_intances0", "prev_track_intances1",
      "prev_track_intances3", "prev_track_intances4",
      "prev_track_intances5", "prev_track_intances6",
      "prev_track_intances8", "prev_track_intances9",
      "prev_track_intances11", "prev_track_intances12",
      "prev_track_intances13"};
  return names;
}

bool engine_has_tensor(const std::shared_ptr<TensorRT::Engine>& engine,
                       const std::string& name) {
  for (int i = 0; i < engine->num_bindings(); ++i) {
    if (engine->get_binding_name(i) == name) return true;
  }
  return false;
}

std::vector<std::string> e2e_output_names(
    const std::shared_ptr<TensorRT::Engine>& engine) {
  std::vector<std::string> names = {
      "prev_track_intances0_out", "prev_track_intances1_out",
      "prev_track_intances3_out", "prev_track_intances4_out",
      "prev_track_intances5_out", "prev_track_intances6_out",
      "prev_track_intances8_out", "prev_track_intances9_out",
      "prev_track_intances11_out", "prev_track_intances12_out",
      "prev_track_intances13_out", "prev_timestamp_out",
      "prev_l2g_t_out", "prev_l2g_r_mat_out", "bev_embed",
      "bboxes_dict_bboxes", "scores", "labels", "bbox_index",
      "obj_idxes", "max_obj_id_out", "traj_scores_0", "traj_0",
      "traj_scores_1", "traj_1", "traj_scores", "traj",
      "valid_traj_masks"};
  if (engine_has_tensor(engine, "sdc_traj")) {
    names.push_back("sdc_traj");
  }
  if (engine_has_tensor(engine, "seg_out")) {
    names.push_back("seg_out");
  }
  return names;
}

struct CliArgs {
  std::string sparse_onnx_path;
  std::string frontend_engine_path;
  std::string e2e_engine_path;
  std::string plugin_path;
  std::string raw_points_input;
  std::string output_dir;
  int num_frames = 0;
  std::vector<int> grid_size = {
      uniad_lidar::kDefaultGridZ,
      uniad_lidar::kDefaultGridY,
      uniad_lidar::kDefaultGridX};
  std::string metadata_input;
  std::string gt_detections_input;
  std::string track_init_dir;
  int track_state_len = 601;
  int max_track_state_len = 901;
  int command = 2;
  bool write_visualization = true;
  uniad_lidar::DetectionDecodeConfig decode_config;
  uniad_lidar::BevVisualizationConfig bev_visualization_config;
};

void usage(const char* program) {
  std::fprintf(
      stderr,
      "Usage: %s <sparse.onnx> <frontend.engine> <e2e.engine> <plugin.so> "
      "<raw_points_file_or_dir_or_pattern> <output_dir> <num_frames> "
      "[grid_z grid_y grid_x] --metadata-json <file_or_dir_or_pattern> "
      "--track-init-dir <dir> [--gt-detections <file_or_dir_or_pattern>] "
      "[--track-state-len <count>] [--max-track-state-len <count>] "
      "[--command <0|1|2>] "
      "[--score-threshold <score>] [--max-dets <count>] "
      "[--bev-max-draw <count>] [--bev-score-threshold <score>] "
      "[--no-visualization]\n",
      program);
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

uniad_lidar::BevGridSpec bev_grid_from_e2e_engine(
    const std::shared_ptr<TensorRT::Engine>& engine) {
  const std::vector<TRT_INT_TYPE> shape = engine->run_dims("prev_bev");
  uniad_lidar::require(
      shape.size() == 4,
      "E2E engine prev_bev must be NCHW, got " +
          uniad_lidar::shape_string_trt(shape));
  for (TRT_INT_TYPE dim : shape) {
    uniad_lidar::require(dim > 0,
                         "E2E engine prev_bev has invalid dimension: " +
                             uniad_lidar::shape_string_trt(shape));
    uniad_lidar::require(
        dim <= static_cast<TRT_INT_TYPE>(std::numeric_limits<int>::max()),
            "E2E engine prev_bev dimension is too large: " +
            uniad_lidar::shape_string_trt(shape));
  }

  uniad_lidar::BevGridSpec grid;
  grid.batch_size = static_cast<int>(shape[0]);
  grid.channels = static_cast<int>(shape[1]);
  grid.height = static_cast<int>(shape[2]);
  grid.width = static_cast<int>(shape[3]);
  uniad_lidar::require(
      grid.batch_size == 1,
      "uniad_lidar_e2e currently supports batch size 1.");
  return grid;
}

std::vector<TRT_INT_TYPE> track_state_shape(
    const std::string& name, int state_len) {
  const TRT_INT_TYPE n = static_cast<TRT_INT_TYPE>(state_len);
  if (name == "prev_track_intances0") return {n, 512};
  if (name == "prev_track_intances1") return {n, 3};
  if (name == "prev_track_intances3") return {n};
  if (name == "prev_track_intances4") return {n};
  if (name == "prev_track_intances5") return {n};
  if (name == "prev_track_intances6") return {n};
  if (name == "prev_track_intances8") return {n};
  if (name == "prev_track_intances9") return {n, 10};
  if (name == "prev_track_intances11") return {n, 4, 256};
  if (name == "prev_track_intances12") return {n, 4};
  if (name == "prev_track_intances13") return {n};
  uniad_lidar::fail("Unknown track state input: " + name);
}

uniad_lidar::TensorMap load_initial_track_state(
    const std::shared_ptr<TensorRT::Engine>& engine,
    const std::string& track_init_dir,
    int track_state_len) {
  uniad_lidar::TensorMap state;
  for (const std::string& name : track_state_input_names()) {
    const std::string path = uniad_lidar::join_path(track_init_dir, name + ".dat");
    uniad_lidar::require(path_exists(path),
                         "Missing initial track state file: " + path);
    state[name] = uniad_lidar::read_raw_tensor(
        path, engine->dtype(name), track_state_shape(name, track_state_len));
  }
  return state;
}

std::vector<float> read_init_float_or_default(
    const std::string& track_init_dir,
    const std::string& name,
    size_t numel,
    const std::vector<float>& fallback) {
  const std::string path = uniad_lidar::join_path(track_init_dir, name + ".dat");
  if (path_exists(path)) {
    return uniad_lidar::read_raw_float(path, numel);
  }
  uniad_lidar::require(fallback.size() == numel,
                       "Invalid fallback size for " + name);
  return fallback;
}

std::vector<int32_t> read_init_int32_or_default(
    const std::string& track_init_dir,
    const std::string& name,
    size_t numel,
    const std::vector<int32_t>& fallback) {
  const std::string path = uniad_lidar::join_path(track_init_dir, name + ".dat");
  if (path_exists(path)) {
    return uniad_lidar::read_raw_int32(path, numel);
  }
  uniad_lidar::require(fallback.size() == numel,
                       "Invalid fallback size for " + name);
  return fallback;
}

std::vector<float> tensor_to_float_vector(
    const std::string& name,
    const uniad_lidar::NamedTensor& tensor) {
  if (!tensor.data.empty()) return tensor.data;
  std::vector<float> out(tensor.int_data.begin(), tensor.int_data.end());
  uniad_lidar::require(!out.empty(), name + " output tensor is empty.");
  return out;
}

std::vector<int32_t> tensor_to_int32_vector(
    const std::string& name,
    const uniad_lidar::NamedTensor& tensor) {
  if (!tensor.int_data.empty()) return tensor.int_data;
  std::vector<int32_t> out(tensor.data.size());
  for (size_t i = 0; i < tensor.data.size(); ++i) {
    out[i] = static_cast<int32_t>(std::lround(tensor.data[i]));
  }
  uniad_lidar::require(!out.empty(), name + " output tensor is empty.");
  return out;
}

std::vector<float> adapt_middle_bev_for_frontend(
    const std::vector<float>& middle_bev,
    size_t expected_numel) {
  if (middle_bev.size() == expected_numel) {
    return middle_bev;
  }
  if (expected_numel > 0 && middle_bev.size() % expected_numel == 0) {
    const size_t batches = middle_bev.size() / expected_numel;
    std::printf("[INFO] sparse output has %zu frontend-sized batches; "
                "using batch 0 for single-frame e2e runtime.\n",
                batches);
    return std::vector<float>(
        middle_bev.begin(), middle_bev.begin() + expected_numel);
  }
  uniad_lidar::fail("Sparse output numel does not match frontend input.");
}

uniad_lidar::NamedTensor coerce_for_track_input(
    const std::shared_ptr<TensorRT::Engine>& engine,
    const std::string& input_name,
    const uniad_lidar::NamedTensor& src) {
  uniad_lidar::NamedTensor out;
  out.shape = src.shape;
  out.dtype = engine->dtype(input_name);
  if (out.dtype == TensorRT::DType::FLOAT) {
    out.data = tensor_to_float_vector(input_name, src);
    return out;
  }
  if (out.dtype == TensorRT::DType::INT32) {
    out.int_data = tensor_to_int32_vector(input_name, src);
    return out;
  }
  uniad_lidar::fail("Unsupported track input dtype for " + input_name);
}

std::string output_name_for_state_input(const std::string& input_name) {
  return input_name + "_out";
}

void metadata_l2g(
    const uniad_lidar::FrameMetadata& metadata,
    std::vector<float>* l2g_r_mat,
    std::vector<float>* l2g_t) {
  uniad_lidar::require(metadata.has_ego2global,
                       "Track runtime requires ego2global in metadata.");
  l2g_r_mat->assign({
      metadata.ego2global[0], metadata.ego2global[1], metadata.ego2global[2],
      metadata.ego2global[4], metadata.ego2global[5], metadata.ego2global[6],
      metadata.ego2global[8], metadata.ego2global[9], metadata.ego2global[10]});
  l2g_t->assign({
      metadata.ego2global[3],
      metadata.ego2global[7],
      metadata.ego2global[11]});
}

CliArgs parse_cli(int argc, char** argv) {
  if (argc < 8 || std::string(argv[1]) == "--help") {
    usage(argv[0]);
    std::exit(argc < 8 ? 2 : 0);
  }

  CliArgs args;
  args.bev_visualization_config.show_track_id = true;
  args.bev_visualization_config.color_by_track_id = true;
  args.sparse_onnx_path = argv[1];
  args.frontend_engine_path = argv[2];
  args.e2e_engine_path = argv[3];
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
    if (option == "--metadata-json") {
      uniad_lidar::require(cursor < argc,
                           "--metadata-json requires a value.");
      args.metadata_input = argv[cursor++];
    } else if (option == "--track-init-dir") {
      uniad_lidar::require(cursor < argc,
                           "--track-init-dir requires a value.");
      args.track_init_dir = argv[cursor++];
    } else if (option == "--gt-detections") {
      uniad_lidar::require(cursor < argc,
                           "--gt-detections requires a value.");
      args.gt_detections_input = argv[cursor++];
    } else if (option == "--track-state-len") {
      uniad_lidar::require(cursor < argc,
                           "--track-state-len requires a value.");
      args.track_state_len = std::atoi(argv[cursor++]);
    } else if (option == "--max-track-state-len") {
      uniad_lidar::require(cursor < argc,
                           "--max-track-state-len requires a value.");
      args.max_track_state_len = std::atoi(argv[cursor++]);
    } else if (option == "--command") {
      uniad_lidar::require(cursor < argc, "--command requires a value.");
      args.command = std::atoi(argv[cursor++]);
    } else if (option == "--no-visualization") {
      args.write_visualization = false;
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
  uniad_lidar::require(!args.metadata_input.empty(),
                       "--metadata-json is required for e2e runtime.");
  uniad_lidar::require(!args.track_init_dir.empty(),
                       "--track-init-dir is required for e2e runtime.");
  uniad_lidar::require(args.track_state_len > 0,
                       "--track-state-len must be positive.");
  uniad_lidar::require(args.max_track_state_len >= args.track_state_len,
                       "--max-track-state-len must be >= --track-state-len.");
  uniad_lidar::require(args.command >= 0 && args.command <= 2,
                       "--command must be 0, 1, or 2.");
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

  std::shared_ptr<TensorRT::Engine> e2e_engine =
      TensorRT::load(args.e2e_engine_path);
  uniad_lidar::require(static_cast<bool>(e2e_engine),
                       "Failed to load LiDAR e2e TensorRT engine.");
  const bool has_planning_input = engine_has_tensor(e2e_engine, "command");
  const std::vector<std::string> output_names = e2e_output_names(e2e_engine);

  const size_t prev_numel =
      static_cast<size_t>(e2e_engine->numel("prev_bev"));
  const uniad_lidar::BevGridSpec prev_bev_grid =
      bev_grid_from_e2e_engine(e2e_engine);

  const uniad_lidar::TensorMap initial_track_state =
      load_initial_track_state(e2e_engine, args.track_init_dir,
                               args.track_state_len);
  uniad_lidar::TensorMap track_state = initial_track_state;

  const std::vector<float> identity_r = {
      1.0f, 0.0f, 0.0f,
      0.0f, 1.0f, 0.0f,
      0.0f, 0.0f, 1.0f};
  const std::vector<float> zero_t = {0.0f, 0.0f, 0.0f};
  const std::vector<float> init_prev_timestamp =
      read_init_float_or_default(args.track_init_dir, "prev_timestamp", 1,
                                 {0.0f});
  const std::vector<float> init_prev_l2g_r_mat =
      read_init_float_or_default(args.track_init_dir, "prev_l2g_r_mat", 9,
                                 identity_r);
  const std::vector<float> init_prev_l2g_t =
      read_init_float_or_default(args.track_init_dir, "prev_l2g_t", 3,
                                 zero_t);
  const std::vector<int32_t> init_max_obj_id =
      read_init_int32_or_default(args.track_init_dir, "max_obj_id", 1,
                                 {0});

  std::vector<float> prev_bev(prev_numel, 0.0f);
  std::vector<float> prev_timestamp = init_prev_timestamp;
  std::vector<float> prev_l2g_r_mat = init_prev_l2g_r_mat;
  std::vector<float> prev_l2g_t = init_prev_l2g_t;
  std::vector<int32_t> max_obj_id = init_max_obj_id;
  bool prev_bev_valid = false;
  bool time_origin_set = false;
  double time_origin = 0.0;

  if (!args.write_visualization && !args.gt_detections_input.empty()) {
    std::printf("[INFO] --no-visualization provided; "
                "--gt-detections will not produce comparison SVGs.\n");
  }

  // Per-stage GPU timing (diagnostic). Sync after each stage so the measured
  // interval is that stage's GPU execution, not just CPU submission. Frame 0
  // excluded from averages (engine warm-up / first-touch allocs).
  auto now_ms = [&]() {
    cudaStreamSynchronize(stream);
    return std::chrono::duration<double, std::milli>(
               std::chrono::steady_clock::now().time_since_epoch())
        .count();
  };
  double t_sparse = 0.0, t_backbone = 0.0, t_dense = 0.0;
  int timed_frames = 0;

  for (int frame = 0; frame < args.num_frames; ++frame) {
    const std::string raw_points_path = resolve_raw_points_path(
        args.raw_points_input, frame, args.num_frames);
    const std::string metadata_path = resolve_metadata_path(
        args.metadata_input, frame, args.num_frames);
    const uniad_lidar::FrameMetadata metadata =
        uniad_lidar::read_frame_metadata(metadata_path);
    uniad_lidar::require(metadata.has_timestamp,
                         "E2E runtime requires timestamp in metadata.");

    const bool metadata_reset =
        metadata.has_prev_bev_exists && !metadata.prev_bev_exists;
    const bool reset_state = !prev_bev_valid || metadata_reset;
    if (reset_state) {
      std::fill(prev_bev.begin(), prev_bev.end(), 0.0f);
      track_state = initial_track_state;
      prev_timestamp = init_prev_timestamp;
      prev_l2g_r_mat = init_prev_l2g_r_mat;
      prev_l2g_t = init_prev_l2g_t;
      max_obj_id = init_max_obj_id;
      time_origin = metadata.timestamp;
      time_origin_set = true;
    } else if (!time_origin_set) {
      time_origin = metadata.timestamp;
      time_origin_set = true;
    }

    std::vector<float> l2g_r_mat;
    std::vector<float> l2g_t;
    metadata_l2g(metadata, &l2g_r_mat, &l2g_t);
    const float relative_timestamp =
        static_cast<float>(metadata.timestamp - time_origin);

    const bool use_prev = prev_bev_valid && !metadata_reset;
    std::vector<float> aligned_prev_bev = prev_bev;
    bool prev_bev_aligned = false;
    if (use_prev) {
      aligned_prev_bev = uniad_lidar::rotate_prev_bev_from_metadata(
          prev_bev, metadata, prev_bev_grid);
      prev_bev_aligned = metadata.has_ego_motion_delta;
    }
    const std::vector<float> shift = uniad_lidar::shift_from_metadata(metadata);
    const int command = metadata.has_command ? metadata.command : args.command;

    std::printf("[INFO] frame %d raw points: %s\n",
                frame, raw_points_path.c_str());
    const char* prev_label = metadata.has_prev_bev_exists
        ? (metadata.prev_bev_exists ? "true" : "false")
        : "missing";
    std::printf("[INFO] frame %d metadata: %s, prev_bev_exists=%s, "
                "use_prev_bev=%d, reset_state=%d, rel_timestamp=%.9g, "
                "shift=(%.9g, %.9g), prev_bev_aligned=%d, command=%d%s\n",
                frame, metadata_path.c_str(), prev_label,
                use_prev ? 1 : 0, reset_state ? 1 : 0,
                relative_timestamp, shift[0], shift[1],
                prev_bev_aligned ? 1 : 0, command,
                metadata.has_command ? "" : " (default)");

    uniad_lidar::SparseInputSpec sparse_input;
    sparse_input.use_raw_points = true;
    sparse_input.raw_points_path = raw_points_path;
    const double ts0 = now_ms();
    std::vector<float> middle_bev = sparse_encoder.forward(
        sparse_input, args.grid_size, stream);
    middle_bev = adapt_middle_bev_for_frontend(
        middle_bev, static_cast<size_t>(frontend_engine->numel("middle_bev")));
    const double ts1 = now_ms();  // end sparse encoder

    std::vector<float> lidar_bev = uniad_lidar::run_single_input_trt(
        frontend_engine,
        "LiDAR Backbone+Neck TensorRT",
        "middle_bev",
        "lidar_bev",
        middle_bev,
        stream,
        frame == 0);
    const double ts2 = now_ms();  // end backbone+neck

    uniad_lidar::TrackLidarInput track_input;
    track_input.lidar_bev = lidar_bev;
    track_input.prev_bev = aligned_prev_bev;
    track_input.shift = shift;
    track_input.use_prev_bev = {use_prev ? 1.0f : 0.0f};
    track_input.prev_timestamp = prev_timestamp;
    track_input.prev_l2g_r_mat = prev_l2g_r_mat;
    track_input.prev_l2g_t = prev_l2g_t;
    track_input.timestamp = {relative_timestamp};
    track_input.l2g_r_mat = l2g_r_mat;
    track_input.l2g_t = l2g_t;
    if (has_planning_input) {
      track_input.command = {static_cast<float>(command)};
    }
    track_input.max_obj_id = max_obj_id;
    track_input.track_state = track_state;

    uniad_lidar::TensorMap outputs = uniad_lidar::run_track_lidar_trt(
        e2e_engine,
        track_input,
        output_names,
        stream,
        "LiDAR E2E TensorRT",
        frame == 0,
        args.max_track_state_len,
        std::max(args.decode_config.max_num, 300));
    const double ts3 = now_ms();  // end dense e2e

    if (frame > 0) {  // skip warm-up frame
      t_sparse += ts1 - ts0;
      t_backbone += ts2 - ts1;
      t_dense += ts3 - ts2;
      ++timed_frames;
    }

    const std::string prefix = output_prefix(args.output_dir, frame);
    uniad_lidar::write_raw_float(prefix + "_lidar_bev.bin", lidar_bev);
    for (const std::string& name : output_names) {
      uniad_lidar::write_raw_tensor(prefix + "_" + name + ".bin",
                                    outputs.at(name));
    }

    const std::vector<uniad_lidar::Detection> detections =
        uniad_lidar::decode_track_detections(
            outputs.at("bboxes_dict_bboxes"),
            outputs.at("scores"),
            outputs.at("labels"),
            outputs.at("obj_idxes"),
            args.decode_config);
    uniad_lidar::write_detections_txt(prefix + "_detections.txt",
                                      detections);
    if (args.write_visualization) {
      uniad_lidar::write_bev_svg(prefix + "_bev.svg",
                                 detections,
                                 args.bev_visualization_config);
    }
    if (args.write_visualization && !args.gt_detections_input.empty()) {
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

    for (const std::string& name : track_state_input_names()) {
      track_state[name] = coerce_for_track_input(
          e2e_engine, name, outputs.at(output_name_for_state_input(name)));
    }
    prev_timestamp = tensor_to_float_vector(
        "prev_timestamp_out", outputs.at("prev_timestamp_out"));
    prev_l2g_t = tensor_to_float_vector(
        "prev_l2g_t_out", outputs.at("prev_l2g_t_out"));
    prev_l2g_r_mat = tensor_to_float_vector(
        "prev_l2g_r_mat_out", outputs.at("prev_l2g_r_mat_out"));
    max_obj_id = tensor_to_int32_vector(
        "max_obj_id_out", outputs.at("max_obj_id_out"));
    prev_bev = outputs.at("bev_embed").data;
    prev_bev_valid = true;

    std::printf("[INFO] frame %d outputs written with prefix %s "
                "(%zu tracks, max_obj_id=%d)\n",
                frame, prefix.c_str(), detections.size(), max_obj_id[0]);
  }

  if (timed_frames > 0) {
    const double s = t_sparse / timed_frames;
    const double b = t_backbone / timed_frames;
    const double d = t_dense / timed_frames;
    const double tot = s + b + d;
    std::printf(
        "\n[STAGE TIMING] avg over %d frames (frame 0 excluded), ms/frame:\n"
        "  sparse encoder : %8.3f  (%5.1f%%)\n"
        "  backbone+neck  : %8.3f  (%5.1f%%)\n"
        "  dense e2e      : %8.3f  (%5.1f%%)\n"
        "  3-stage total  : %8.3f\n",
        timed_frames, s, 100.0 * s / tot, b, 100.0 * b / tot,
        d, 100.0 * d / tot, tot);
  }

  uniad_lidar::check_cuda(cudaStreamDestroy(stream), "cudaStreamDestroy");
  return 0;
}
