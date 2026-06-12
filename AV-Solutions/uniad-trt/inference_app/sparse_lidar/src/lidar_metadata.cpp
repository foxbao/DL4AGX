/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: MIT
 */

#include "lidar_metadata.hpp"

#include <algorithm>
#include <cctype>
#include <cmath>
#include <cstdlib>
#include <fstream>
#include <map>
#include <memory>
#include <sstream>
#include <string>
#include <utility>

#include "lidar_runtime.hpp"

namespace uniad_lidar {
namespace {

enum class JsonType { Null, Bool, Number, String, Array, Object };

struct JsonValue {
  JsonType type = JsonType::Null;
  bool bool_value = false;
  double number_value = 0.0;
  std::string string_value;
  std::vector<JsonValue> array_value;
  std::map<std::string, JsonValue> object_value;

  const JsonValue* get(const std::string& key) const {
    if (type != JsonType::Object) return nullptr;
    auto iter = object_value.find(key);
    return iter == object_value.end() ? nullptr : &iter->second;
  }
};

class JsonParser {
 public:
  explicit JsonParser(const std::string& text) : text_(text) {}

  JsonValue parse() {
    JsonValue value = parse_value();
    skip_ws();
    require(pos_ == text_.size(), "Unexpected trailing JSON content.");
    return value;
  }

 private:
  void skip_ws() {
    while (pos_ < text_.size() &&
           std::isspace(static_cast<unsigned char>(text_[pos_]))) {
      ++pos_;
    }
  }

  char peek() {
    skip_ws();
    require(pos_ < text_.size(), "Unexpected end of JSON.");
    return text_[pos_];
  }

  bool consume(char expected) {
    skip_ws();
    if (pos_ < text_.size() && text_[pos_] == expected) {
      ++pos_;
      return true;
    }
    return false;
  }

  void expect(char expected) {
    require(consume(expected),
            std::string("Expected JSON character: ") + expected);
  }

  JsonValue parse_value() {
    const char c = peek();
    if (c == 'n') return parse_literal("null", JsonValue{});
    if (c == 't') {
      JsonValue value = parse_literal("true", JsonValue{});
      value.type = JsonType::Bool;
      value.bool_value = true;
      return value;
    }
    if (c == 'f') {
      JsonValue value = parse_literal("false", JsonValue{});
      value.type = JsonType::Bool;
      value.bool_value = false;
      return value;
    }
    if (c == '"') return parse_string();
    if (c == '[') return parse_array();
    if (c == '{') return parse_object();
    return parse_number();
  }

  JsonValue parse_literal(const char* literal, JsonValue value) {
    const size_t len = std::char_traits<char>::length(literal);
    require(text_.compare(pos_, len, literal) == 0,
            std::string("Expected JSON literal: ") + literal);
    pos_ += len;
    return value;
  }

  JsonValue parse_string() {
    expect('"');
    std::string out;
    while (pos_ < text_.size()) {
      const char c = text_[pos_++];
      if (c == '"') {
        JsonValue value;
        value.type = JsonType::String;
        value.string_value = std::move(out);
        return value;
      }
      if (c == '\\') {
        require(pos_ < text_.size(), "Unterminated JSON escape.");
        const char esc = text_[pos_++];
        switch (esc) {
          case '"':
          case '\\':
          case '/':
            out.push_back(esc);
            break;
          case 'b':
            out.push_back('\b');
            break;
          case 'f':
            out.push_back('\f');
            break;
          case 'n':
            out.push_back('\n');
            break;
          case 'r':
            out.push_back('\r');
            break;
          case 't':
            out.push_back('\t');
            break;
          case 'u':
            require(pos_ + 4 <= text_.size(),
                    "Incomplete JSON unicode escape.");
            out.push_back('?');
            pos_ += 4;
            break;
          default:
            fail("Unsupported JSON escape sequence.");
        }
      } else {
        out.push_back(c);
      }
    }
    fail("Unterminated JSON string.");
  }

  JsonValue parse_number() {
    skip_ws();
    const char* begin = text_.c_str() + pos_;
    char* end = nullptr;
    const double value = std::strtod(begin, &end);
    require(end != begin, "Expected JSON number.");
    pos_ = static_cast<size_t>(end - text_.c_str());
    JsonValue out;
    out.type = JsonType::Number;
    out.number_value = value;
    return out;
  }

  JsonValue parse_array() {
    expect('[');
    JsonValue out;
    out.type = JsonType::Array;
    if (consume(']')) return out;
    while (true) {
      out.array_value.push_back(parse_value());
      if (consume(']')) return out;
      expect(',');
    }
  }

  JsonValue parse_object() {
    expect('{');
    JsonValue out;
    out.type = JsonType::Object;
    if (consume('}')) return out;
    while (true) {
      JsonValue key = parse_string();
      expect(':');
      out.object_value[key.string_value] = parse_value();
      if (consume('}')) return out;
      expect(',');
    }
  }

  const std::string& text_;
  size_t pos_ = 0;
};

std::string read_text_file(const std::string& path) {
  std::ifstream in(path);
  require(static_cast<bool>(in), "Failed to open metadata JSON: " + path);
  std::ostringstream ss;
  ss << in.rdbuf();
  return ss.str();
}

bool numeric_key_less(const std::string& a, const std::string& b) {
  char* end_a = nullptr;
  char* end_b = nullptr;
  const long va = std::strtol(a.c_str(), &end_a, 10);
  const long vb = std::strtol(b.c_str(), &end_b, 10);
  const bool a_numeric = end_a != a.c_str() && *end_a == '\0';
  const bool b_numeric = end_b != b.c_str() && *end_b == '\0';
  if (a_numeric && b_numeric) return va < vb;
  return a < b;
}

const JsonValue* select_current_meta(const JsonValue& root) {
  const JsonValue* value = &root;
  if (value->type == JsonType::Array) {
    require(!value->array_value.empty(), "Metadata JSON array is empty.");
    value = &value->array_value.front();
  }

  const JsonValue* queue_metas = value->get("queue_metas");
  if (queue_metas != nullptr) {
    value = queue_metas;
  }

  if (value->type == JsonType::Object && value->get("ego_motion_delta")) {
    return value;
  }

  require(value->type == JsonType::Object,
          "Metadata JSON does not contain an object with ego_motion_delta.");
  require(!value->object_value.empty(), "queue_metas object is empty.");
  auto current = value->object_value.begin();
  for (auto iter = value->object_value.begin();
       iter != value->object_value.end(); ++iter) {
    if (numeric_key_less(current->first, iter->first)) {
      current = iter;
    }
  }
  return &current->second;
}

void parse_matrix4x4_into(
    const JsonValue& value,
    const char* field_name,
    float* output) {
  require(value.type == JsonType::Array && value.array_value.size() == 4,
          std::string(field_name) + " must be a 4x4 JSON array.");
  for (size_t row = 0; row < 4; ++row) {
    const JsonValue& row_value = value.array_value[row];
    require(row_value.type == JsonType::Array &&
                row_value.array_value.size() == 4,
            std::string(field_name) + " must be a 4x4 JSON array.");
    for (size_t col = 0; col < 4; ++col) {
      const JsonValue& number = row_value.array_value[col];
      require(number.type == JsonType::Number,
              std::string(field_name) + " elements must be numbers.");
      output[row * 4 + col] = static_cast<float>(number.number_value);
    }
  }
}

void parse_ego_motion_delta(const JsonValue& value, FrameMetadata* metadata) {
  parse_matrix4x4_into(
      value, "ego_motion_delta", metadata->ego_motion_delta);
  metadata->has_ego_motion_delta = true;
}

void parse_ego2global(const JsonValue& value, FrameMetadata* metadata) {
  parse_matrix4x4_into(value, "ego2global", metadata->ego2global);
  metadata->has_ego2global = true;
}

struct Rotation2D {
  float r00 = 1.0f;
  float r01 = 0.0f;
  float r10 = 0.0f;
  float r11 = 1.0f;
  bool is_identity = true;
};

float determinant3x3(const float* m) {
  return m[0] * (m[5] * m[10] - m[6] * m[9]) -
         m[1] * (m[4] * m[10] - m[6] * m[8]) +
         m[2] * (m[4] * m[9] - m[5] * m[8]);
}

Rotation2D prev_from_current_rotation(const FrameMetadata& metadata) {
  Rotation2D rotation;
  if (!metadata.has_ego_motion_delta ||
      (metadata.has_prev_bev_exists && !metadata.prev_bev_exists)) {
    return rotation;
  }

  const float* m = metadata.ego_motion_delta;
  const float det = determinant3x3(m);
  if (std::fabs(det) < 1e-6f) {
    return rotation;
  }

  rotation.r00 = (m[5] * m[10] - m[6] * m[9]) / det;
  rotation.r01 = (m[2] * m[9] - m[1] * m[10]) / det;
  rotation.r10 = (m[6] * m[8] - m[4] * m[10]) / det;
  rotation.r11 = (m[0] * m[10] - m[2] * m[8]) / det;
  rotation.is_identity =
      std::fabs(rotation.r00 - 1.0f) < 1e-7f &&
      std::fabs(rotation.r01) < 1e-7f &&
      std::fabs(rotation.r10) < 1e-7f &&
      std::fabs(rotation.r11 - 1.0f) < 1e-7f;
  return rotation;
}

struct BilinearSample {
  int i00 = -1;
  int i01 = -1;
  int i10 = -1;
  int i11 = -1;
  float w00 = 0.0f;
  float w01 = 0.0f;
  float w10 = 0.0f;
  float w11 = 0.0f;
};

int spatial_index_or_invalid(int y, int x, int height, int width) {
  if (x < 0 || x >= width || y < 0 || y >= height) return -1;
  return y * width + x;
}

}  // namespace

FrameMetadata read_frame_metadata(const std::string& path) {
  const JsonValue root = JsonParser(read_text_file(path)).parse();
  const JsonValue* current = select_current_meta(root);
  require(current != nullptr && current->type == JsonType::Object,
          "Failed to select current metadata object.");

  FrameMetadata metadata;
  if (const JsonValue* prev = current->get("prev_bev_exists")) {
    require(prev->type == JsonType::Bool,
            "prev_bev_exists must be a JSON boolean.");
    metadata.has_prev_bev_exists = true;
    metadata.prev_bev_exists = prev->bool_value;
  }
  if (const JsonValue* delta = current->get("ego_motion_delta")) {
    parse_ego_motion_delta(*delta, &metadata);
  }
  if (const JsonValue* timestamp = current->get("timestamp")) {
    require(timestamp->type == JsonType::Number,
            "timestamp must be a JSON number.");
    metadata.has_timestamp = true;
    metadata.timestamp = timestamp->number_value;
  }
  if (const JsonValue* ego2global = current->get("ego2global")) {
    parse_ego2global(*ego2global, &metadata);
  }
  return metadata;
}

std::vector<float> shift_from_metadata(
    const FrameMetadata& metadata,
    float x_extent,
    float y_extent,
    bool rotate_prev_bev) {
  if (!metadata.has_ego_motion_delta ||
      (metadata.has_prev_bev_exists && !metadata.prev_bev_exists)) {
    return {0.0f, 0.0f};
  }

  require(std::fabs(x_extent) > 1e-6f && std::fabs(y_extent) > 1e-6f,
          "Point cloud extents must be non-zero.");
  if (rotate_prev_bev) {
    return {
        -metadata.ego_motion_delta[3] / x_extent,
        -metadata.ego_motion_delta[7] / y_extent};
  }

  fail("Metadata shift without rotate_prev_bev is not implemented.");
}

std::vector<float> rotate_prev_bev_from_metadata(
    const std::vector<float>& prev_bev,
    const FrameMetadata& metadata,
    const BevGridSpec& grid) {
  require(grid.batch_size > 0 && grid.channels > 0 &&
              grid.height > 0 && grid.width > 0,
          "Invalid BEV grid shape.");
  require(grid.batch_size == 1,
          "LiDAR prev_bev metadata rotation currently supports batch size 1.");
  require(grid.x_max > grid.x_min && grid.y_max > grid.y_min,
          "Invalid BEV point cloud range.");

  const size_t spatial =
      static_cast<size_t>(grid.height) * static_cast<size_t>(grid.width);
  const size_t expected =
      static_cast<size_t>(grid.batch_size) *
      static_cast<size_t>(grid.channels) * spatial;
  require(prev_bev.size() == expected,
          "prev_bev size does not match BEV grid shape.");

  const Rotation2D rotation = prev_from_current_rotation(metadata);
  if (rotation.is_identity) {
    return prev_bev;
  }

  const float x_extent = grid.x_max - grid.x_min;
  const float y_extent = grid.y_max - grid.y_min;
  const float cell_x = x_extent / static_cast<float>(grid.width);
  const float cell_y = y_extent / static_cast<float>(grid.height);

  std::vector<BilinearSample> samples(spatial);
  for (int y = 0; y < grid.height; ++y) {
    const float world_y = grid.y_min + (static_cast<float>(y) + 0.5f) * cell_y;
    for (int x = 0; x < grid.width; ++x) {
      const float world_x =
          grid.x_min + (static_cast<float>(x) + 0.5f) * cell_x;
      const float src_x = rotation.r00 * world_x + rotation.r01 * world_y;
      const float src_y = rotation.r10 * world_x + rotation.r11 * world_y;
      const float ix = (src_x - grid.x_min) / x_extent *
                           static_cast<float>(grid.width) -
                       0.5f;
      const float iy = (src_y - grid.y_min) / y_extent *
                           static_cast<float>(grid.height) -
                       0.5f;

      const int x0 = static_cast<int>(std::floor(ix));
      const int y0 = static_cast<int>(std::floor(iy));
      const int x1 = x0 + 1;
      const int y1 = y0 + 1;
      const float wx = ix - static_cast<float>(x0);
      const float wy = iy - static_cast<float>(y0);

      BilinearSample sample;
      sample.i00 = spatial_index_or_invalid(y0, x0, grid.height, grid.width);
      sample.i01 = spatial_index_or_invalid(y0, x1, grid.height, grid.width);
      sample.i10 = spatial_index_or_invalid(y1, x0, grid.height, grid.width);
      sample.i11 = spatial_index_or_invalid(y1, x1, grid.height, grid.width);
      sample.w00 = (1.0f - wx) * (1.0f - wy);
      sample.w01 = wx * (1.0f - wy);
      sample.w10 = (1.0f - wx) * wy;
      sample.w11 = wx * wy;
      samples[static_cast<size_t>(y) * static_cast<size_t>(grid.width) +
              static_cast<size_t>(x)] = sample;
    }
  }

  std::vector<float> rotated(prev_bev.size(), 0.0f);
  for (int batch = 0; batch < grid.batch_size; ++batch) {
    for (int channel = 0; channel < grid.channels; ++channel) {
      const size_t base =
          (static_cast<size_t>(batch) * static_cast<size_t>(grid.channels) +
           static_cast<size_t>(channel)) * spatial;
      for (size_t i = 0; i < spatial; ++i) {
        const BilinearSample& sample = samples[i];
        float value = 0.0f;
        if (sample.i00 >= 0) {
          value += sample.w00 * prev_bev[base + static_cast<size_t>(sample.i00)];
        }
        if (sample.i01 >= 0) {
          value += sample.w01 * prev_bev[base + static_cast<size_t>(sample.i01)];
        }
        if (sample.i10 >= 0) {
          value += sample.w10 * prev_bev[base + static_cast<size_t>(sample.i10)];
        }
        if (sample.i11 >= 0) {
          value += sample.w11 * prev_bev[base + static_cast<size_t>(sample.i11)];
        }
        rotated[base + i] = value;
      }
    }
  }
  return rotated;
}

}  // namespace uniad_lidar
