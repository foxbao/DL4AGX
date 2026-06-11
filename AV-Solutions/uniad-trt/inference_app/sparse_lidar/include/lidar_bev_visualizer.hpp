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
  float min_score = -1.0f;
  std::array<float, 4> xy_range = {{-64.0f, -48.0f, 64.0f, 48.0f}};
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

}  // namespace uniad_lidar

#endif  // UNIAD_SPARSE_LIDAR_BEV_VISUALIZER_HPP_
