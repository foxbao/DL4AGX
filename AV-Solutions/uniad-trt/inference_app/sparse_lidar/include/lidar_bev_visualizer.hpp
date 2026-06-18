/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: MIT
 */

#ifndef UNIAD_SPARSE_LIDAR_BEV_VISUALIZER_HPP_
#define UNIAD_SPARSE_LIDAR_BEV_VISUALIZER_HPP_

#include <array>
#include <string>
#include <vector>

#include "lidar_postprocess.hpp"

namespace uniad_lidar {

struct BevVisualizationConfig {
  int image_width = 900;
  int image_height = 1200;
  int max_draw = 100;
  int max_points = 30000;
  float min_score = -1.0f;
  bool show_track_id = false;
  bool color_by_track_id = false;
  std::array<float, 4> xy_range = {{-64.0f, -48.0f, 64.0f, 48.0f}};
};

struct BevPoint {
  float x = 0.0f;
  float y = 0.0f;
};

struct DrivableScoreMap {
  std::vector<float> data;
  int height = 0;
  int width = 0;
  float threshold = 0.5f;
};

void write_bev_svg(
    const std::string& path,
    const std::vector<Detection>& detections,
    const BevVisualizationConfig& config = BevVisualizationConfig());

void write_bev_comparison_svg(
    const std::string& path,
    const std::vector<Detection>& gt_detections,
    const std::vector<Detection>& pred_detections,
    const BevVisualizationConfig& config = BevVisualizationConfig());

void write_bev_drivable_svg(
    const std::string& path,
    const std::vector<Detection>& detections,
    const DrivableScoreMap& drivable,
    const std::vector<BevPoint>& points,
    const BevVisualizationConfig& config = BevVisualizationConfig());

}  // namespace uniad_lidar

#endif  // UNIAD_SPARSE_LIDAR_BEV_VISUALIZER_HPP_
