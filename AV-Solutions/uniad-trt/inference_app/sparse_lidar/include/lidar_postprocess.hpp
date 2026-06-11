/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: MIT
 */

#ifndef UNIAD_SPARSE_LIDAR_POSTPROCESS_HPP_
#define UNIAD_SPARSE_LIDAR_POSTPROCESS_HPP_

#include <array>
#include <string>
#include <vector>

#include "tensorrt.hpp"

namespace uniad_lidar {

struct DetectionDecodeConfig {
  int max_num = 300;
  int batch_index = 0;
  int decoder_layer = -1;
  float score_threshold = -1.0f;
  std::array<float, 6> post_center_range = {
      {-64.0f, -48.0f, -10.0f, 64.0f, 48.0f, 10.0f}};
};

struct Detection {
  float x = 0.0f;
  float y = 0.0f;
  float z = 0.0f;
  float length = 0.0f;
  float width = 0.0f;
  float height = 0.0f;
  float yaw = 0.0f;
  float vx = 0.0f;
  float vy = 0.0f;
  float score = 0.0f;
  int label = 0;
  int query_index = 0;
};

std::vector<Detection> decode_lidar_detections(
    const std::vector<float>& all_cls_scores,
    const std::vector<TRT_INT_TYPE>& cls_shape,
    const std::vector<float>& all_bbox_preds,
    const std::vector<TRT_INT_TYPE>& bbox_shape,
    const DetectionDecodeConfig& config = DetectionDecodeConfig());

void write_detections_txt(
    const std::string& path,
    const std::vector<Detection>& detections);

std::vector<Detection> read_detections_txt(const std::string& path);

}  // namespace uniad_lidar

#endif  // UNIAD_SPARSE_LIDAR_POSTPROCESS_HPP_
