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

void parse_matrix4x4(const JsonValue& value, FrameMetadata* metadata) {
  require(value.type == JsonType::Array && value.array_value.size() == 4,
          "ego_motion_delta must be a 4x4 JSON array.");
  for (size_t row = 0; row < 4; ++row) {
    const JsonValue& row_value = value.array_value[row];
    require(row_value.type == JsonType::Array &&
                row_value.array_value.size() == 4,
            "ego_motion_delta must be a 4x4 JSON array.");
    for (size_t col = 0; col < 4; ++col) {
      const JsonValue& number = row_value.array_value[col];
      require(number.type == JsonType::Number,
              "ego_motion_delta elements must be numbers.");
      metadata->ego_motion_delta[row * 4 + col] =
          static_cast<float>(number.number_value);
    }
  }
  metadata->has_ego_motion_delta = true;
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
    parse_matrix4x4(*delta, &metadata);
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

}  // namespace uniad_lidar
