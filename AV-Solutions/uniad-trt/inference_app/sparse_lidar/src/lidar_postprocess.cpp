/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: MIT
 */

#include "lidar_postprocess.hpp"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <fstream>
#include <sstream>

#include "lidar_runtime.hpp"

namespace uniad_lidar {
namespace {

struct ScoredIndex {
  float score = 0.0f;
  int flat_index = 0;
};

float sigmoid(float value) {
  if (value >= 0.0f) {
    const float exp_neg = std::exp(-value);
    return 1.0f / (1.0f + exp_neg);
  }
  const float exp_pos = std::exp(value);
  return exp_pos / (1.0f + exp_pos);
}

bool center_in_range(
    const Detection& det,
    const std::array<float, 6>& range) {
  return det.x >= range[0] && det.y >= range[1] && det.z >= range[2] &&
         det.x <= range[3] && det.y <= range[4] && det.z <= range[5];
}

int tensor_int_at(const NamedTensor& tensor, size_t index) {
  if (!tensor.int_data.empty()) {
    require(index < tensor.int_data.size(), "int tensor index out of range.");
    return tensor.int_data[index];
  }
  require(index < tensor.data.size(), "float tensor index out of range.");
  return static_cast<int>(std::lround(tensor.data[index]));
}

}  // namespace

std::vector<Detection> decode_lidar_detections(
    const std::vector<float>& all_cls_scores,
    const std::vector<TRT_INT_TYPE>& cls_shape,
    const std::vector<float>& all_bbox_preds,
    const std::vector<TRT_INT_TYPE>& bbox_shape,
    const DetectionDecodeConfig& config) {
  require(cls_shape.size() == 4,
          "all_cls_scores shape must be [layers, batch, query, classes].");
  require(bbox_shape.size() == 4,
          "all_bbox_preds shape must be [layers, batch, query, code].");
  require(cls_shape[0] == bbox_shape[0] && cls_shape[1] == bbox_shape[1] &&
              cls_shape[2] == bbox_shape[2],
          "Classification and bbox output shapes do not match.");
  require(bbox_shape[3] >= 8,
          "all_bbox_preds code dimension must be at least 8.");

  const int num_layers = static_cast<int>(cls_shape[0]);
  const int batch_size = static_cast<int>(cls_shape[1]);
  const int num_query = static_cast<int>(cls_shape[2]);
  const int num_classes = static_cast<int>(cls_shape[3]);
  const int code_size = static_cast<int>(bbox_shape[3]);
  const int layer =
      config.decoder_layer < 0 ? num_layers - 1 : config.decoder_layer;
  require(layer >= 0 && layer < num_layers, "Invalid decoder layer index.");
  require(config.batch_index >= 0 && config.batch_index < batch_size,
          "Invalid batch index.");

  const size_t expected_cls_numel =
      static_cast<size_t>(num_layers) * batch_size * num_query * num_classes;
  const size_t expected_bbox_numel =
      static_cast<size_t>(num_layers) * batch_size * num_query * code_size;
  require(all_cls_scores.size() == expected_cls_numel,
          "all_cls_scores data size does not match its shape.");
  require(all_bbox_preds.size() == expected_bbox_numel,
          "all_bbox_preds data size does not match its shape.");

  std::vector<ScoredIndex> scored;
  scored.reserve(static_cast<size_t>(num_query) * num_classes);
  const size_t cls_base =
      (static_cast<size_t>(layer) * batch_size + config.batch_index) *
      num_query * num_classes;
  for (int query = 0; query < num_query; ++query) {
    for (int cls = 0; cls < num_classes; ++cls) {
      const int flat_index = query * num_classes + cls;
      scored.push_back({
          sigmoid(all_cls_scores[cls_base + flat_index]),
          flat_index});
    }
  }

  const int max_num = std::min(std::max(config.max_num, 0), num_query);
  if (max_num == 0) return {};
  std::partial_sort(
      scored.begin(), scored.begin() + max_num, scored.end(),
      [](const ScoredIndex& a, const ScoredIndex& b) {
        return a.score > b.score;
      });

  std::vector<Detection> detections;
  detections.reserve(max_num);
  const size_t bbox_base =
      (static_cast<size_t>(layer) * batch_size + config.batch_index) *
      num_query * code_size;
  for (int i = 0; i < max_num; ++i) {
    const float score = scored[i].score;
    if (config.score_threshold >= 0.0f && score <= config.score_threshold) {
      continue;
    }

    const int label = scored[i].flat_index % num_classes;
    const int query = scored[i].flat_index / num_classes;
    const float* pred = all_bbox_preds.data() + bbox_base +
                        static_cast<size_t>(query) * code_size;

    Detection det;
    det.x = pred[0];
    det.y = pred[1];
    det.z = pred[4];
    det.length = std::exp(pred[2]);
    det.width = std::exp(pred[3]);
    det.height = std::exp(pred[5]);
    det.yaw = std::atan2(pred[6], pred[7]);
    if (code_size > 8) {
      det.vx = pred[8];
      det.vy = pred[9];
    }
    det.score = score;
    det.label = label;
    det.query_index = query;

    if (center_in_range(det, config.post_center_range)) {
      detections.push_back(det);
    }
  }
  return detections;
}

std::vector<Detection> decode_track_detections(
    const NamedTensor& bboxes,
    const NamedTensor& scores,
    const NamedTensor& labels,
    const NamedTensor& obj_idxes,
    const DetectionDecodeConfig& config) {
  require(bboxes.shape.size() == 2,
          "bboxes_dict_bboxes shape must be [num_dets, box_code].");
  require(scores.shape.size() == 1, "scores shape must be [num_dets].");
  require(labels.shape.size() == 1, "labels shape must be [num_dets].");
  require(obj_idxes.shape.size() == 1, "obj_idxes shape must be [num_dets].");
  require(bboxes.shape[1] >= 7,
          "bboxes_dict_bboxes code dimension must be at least 7.");

  const int num_dets = static_cast<int>(bboxes.shape[0]);
  const int code_size = static_cast<int>(bboxes.shape[1]);
  require(scores.shape[0] == num_dets && labels.shape[0] == num_dets &&
              obj_idxes.shape[0] == num_dets,
          "Track detection output shapes do not match.");
  require(bboxes.data.size() ==
              static_cast<size_t>(num_dets) * static_cast<size_t>(code_size),
          "bboxes_dict_bboxes data size does not match its shape.");
  require(scores.data.size() == static_cast<size_t>(num_dets),
          "scores data size does not match its shape.");

  const int limit = std::min(num_dets, std::max(config.max_num, 0));
  std::vector<Detection> detections;
  detections.reserve(static_cast<size_t>(limit));
  for (int i = 0; i < limit; ++i) {
    const float score = scores.data[static_cast<size_t>(i)];
    if (config.score_threshold >= 0.0f && score <= config.score_threshold) {
      continue;
    }
    const float* box = bboxes.data.data() + static_cast<size_t>(i) * code_size;

    Detection det;
    det.x = box[0];
    det.y = box[1];
    det.z = box[2];
    det.length = box[3];
    det.width = box[4];
    det.height = box[5];
    det.yaw = box[6];
    if (code_size > 8) {
      det.vx = box[7];
      det.vy = box[8];
    }
    det.score = score;
    det.label = tensor_int_at(labels, static_cast<size_t>(i));
    det.query_index = tensor_int_at(obj_idxes, static_cast<size_t>(i));

    if (center_in_range(det, config.post_center_range)) {
      detections.push_back(det);
    }
  }
  return detections;
}

void write_detections_txt(
    const std::string& path,
    const std::vector<Detection>& detections) {
  std::ofstream out(path);
  require(static_cast<bool>(out), "Failed to open detection output: " + path);
  out << "# x y z length width height yaw vx vy label score query_index\n";
  out.setf(std::ios::fixed);
  out.precision(6);
  for (const Detection& det : detections) {
    out << det.x << ' ' << det.y << ' ' << det.z << ' '
        << det.length << ' ' << det.width << ' ' << det.height << ' '
        << det.yaw << ' ' << det.vx << ' ' << det.vy << ' '
        << det.label << ' ';
    out.precision(9);
    out << det.score << ' ';
    out.precision(6);
    out << det.query_index << '\n';
  }
  require(static_cast<bool>(out), "Failed to write detection output: " + path);
}

std::vector<Detection> read_detections_txt(const std::string& path) {
  std::ifstream in(path);
  require(static_cast<bool>(in), "Failed to open detection input: " + path);

  std::vector<Detection> detections;
  std::string line;
  while (std::getline(in, line)) {
    if (line.empty() || line[0] == '#') continue;

    Detection det;
    std::istringstream ss(line);
    ss >> det.x >> det.y >> det.z
       >> det.length >> det.width >> det.height
       >> det.yaw >> det.vx >> det.vy
       >> det.label >> det.score >> det.query_index;
    require(!ss.fail(), "Malformed detection line in: " + path);
    detections.push_back(det);
  }
  return detections;
}

}  // namespace uniad_lidar
