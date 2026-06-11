/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: MIT
 */

#ifndef UNIAD_SPARSE_LIDAR_METADATA_HPP_
#define UNIAD_SPARSE_LIDAR_METADATA_HPP_

#include <string>
#include <vector>

namespace uniad_lidar {

struct FrameMetadata {
  bool has_prev_bev_exists = false;
  bool prev_bev_exists = false;
  bool has_ego_motion_delta = false;
  float ego_motion_delta[16] = {
      1.0f, 0.0f, 0.0f, 0.0f,
      0.0f, 1.0f, 0.0f, 0.0f,
      0.0f, 0.0f, 1.0f, 0.0f,
      0.0f, 0.0f, 0.0f, 1.0f};
};

struct BevGridSpec {
  int batch_size = 1;
  int channels = 256;
  int height = 120;
  int width = 160;
  float x_min = -64.0f;
  float y_min = -48.0f;
  float x_max = 64.0f;
  float y_max = 48.0f;
};

FrameMetadata read_frame_metadata(const std::string& path);

std::vector<float> shift_from_metadata(
    const FrameMetadata& metadata,
    float x_extent = 128.0f,
    float y_extent = 96.0f,
    bool rotate_prev_bev = true);

std::vector<float> rotate_prev_bev_from_metadata(
    const std::vector<float>& prev_bev,
    const FrameMetadata& metadata,
    const BevGridSpec& grid = BevGridSpec{});

}  // namespace uniad_lidar

#endif  // UNIAD_SPARSE_LIDAR_METADATA_HPP_
