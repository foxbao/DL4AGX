/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: MIT
 */

#include "lidar_bev_visualizer.hpp"

#include <algorithm>
#include <cmath>
#include <fstream>
#include <iomanip>
#include <sstream>
#include <string>

#include "lidar_runtime.hpp"

namespace uniad_lidar {
namespace {

struct PixelPoint {
  float x = 0.0f;
  float y = 0.0f;
};

struct Color {
  int r = 255;
  int g = 255;
  int b = 255;
};

const Color kPalette[] = {
    {0, 114, 189},   {217, 83, 25},   {237, 177, 32},
    {126, 47, 142},  {119, 172, 48},  {77, 190, 238},
    {162, 20, 47},   {51, 160, 44},   {166, 86, 40},
    {255, 127, 0},   {106, 61, 154},  {31, 120, 180},
    {227, 26, 28},
};

std::string color_string(const Color& color) {
  std::ostringstream out;
  out << "rgb(" << color.r << "," << color.g << "," << color.b << ")";
  return out.str();
}

size_t palette_index(int value) {
  constexpr size_t kPaletteSize = sizeof(kPalette) / sizeof(kPalette[0]);
  int index = value % static_cast<int>(kPaletteSize);
  if (index < 0) index += static_cast<int>(kPaletteSize);
  return static_cast<size_t>(index);
}

float scale_for(const BevVisualizationConfig& config) {
  const float x_min = config.xy_range[0];
  const float y_min = config.xy_range[1];
  const float x_max = config.xy_range[2];
  const float y_max = config.xy_range[3];
  const float sx = static_cast<float>(config.image_height) / (x_max - x_min);
  const float sy = static_cast<float>(config.image_width) / (y_max - y_min);
  return std::min(sx, sy);
}

PixelPoint world_to_pixel(
    float x,
    float y,
    const BevVisualizationConfig& config) {
  const float x_min = config.xy_range[0];
  const float y_min = config.xy_range[1];
  const float x_max = config.xy_range[2];
  const float y_max = config.xy_range[3];
  const float scale = scale_for(config);
  const float used_width = (y_max - y_min) * scale;
  const float used_height = (x_max - x_min) * scale;
  const float offset_x =
      (static_cast<float>(config.image_width) - used_width) * 0.5f;
  const float offset_y =
      (static_cast<float>(config.image_height) - used_height) * 0.5f;
  PixelPoint point;
  point.x = offset_x + (y - y_min) * scale;
  point.y = offset_y + used_height - (x - x_min) * scale;
  return point;
}

std::vector<PixelPoint> box_corners(
    const Detection& det,
    const BevVisualizationConfig& config) {
  const float c = std::cos(det.yaw);
  const float s = std::sin(det.yaw);
  const float half_l = det.length * 0.5f;
  const float half_w = det.width * 0.5f;
  const float local[4][2] = {
      {-half_l, -half_w},
      {half_l, -half_w},
      {half_l, half_w},
      {-half_l, half_w},
  };

  std::vector<PixelPoint> corners;
  corners.reserve(4);
  for (const auto& xy : local) {
    const float x = det.x + xy[0] * c - xy[1] * s;
    const float y = det.y + xy[0] * s + xy[1] * c;
    corners.push_back(world_to_pixel(x, y, config));
  }
  return corners;
}

void write_polygon_points(
    std::ostream& out,
    const std::vector<PixelPoint>& points) {
  for (size_t i = 0; i < points.size(); ++i) {
    if (i > 0) out << ' ';
    out << std::fixed << std::setprecision(2)
        << points[i].x << ',' << points[i].y;
  }
}

void write_grid(
    std::ostream& out,
    const BevVisualizationConfig& config) {
  const float scale = scale_for(config);
  const PixelPoint ego = world_to_pixel(0.0f, 0.0f, config);

  out << "<g stroke=\"#2d3741\" stroke-width=\"1\" fill=\"none\">\n";
  for (int x = -60; x <= 60; x += 10) {
    const PixelPoint a = world_to_pixel(static_cast<float>(x),
                                        config.xy_range[1], config);
    const PixelPoint b = world_to_pixel(static_cast<float>(x),
                                        config.xy_range[3], config);
    out << "<line x1=\"" << a.x << "\" y1=\"" << a.y
        << "\" x2=\"" << b.x << "\" y2=\"" << b.y << "\"/>\n";
  }
  for (int y = -40; y <= 40; y += 10) {
    const PixelPoint a = world_to_pixel(config.xy_range[0],
                                        static_cast<float>(y), config);
    const PixelPoint b = world_to_pixel(config.xy_range[2],
                                        static_cast<float>(y), config);
    out << "<line x1=\"" << a.x << "\" y1=\"" << a.y
        << "\" x2=\"" << b.x << "\" y2=\"" << b.y << "\"/>\n";
  }
  for (int radius = 15; radius <= 60; radius += 15) {
    out << "<circle cx=\"" << ego.x << "\" cy=\"" << ego.y
        << "\" r=\"" << radius * scale << "\"/>\n";
  }
  out << "</g>\n";
}

void write_ego(std::ostream& out, const BevVisualizationConfig& config) {
  const PixelPoint ego = world_to_pixel(0.0f, 0.0f, config);
  out << "<g fill=\"none\" stroke=\"#f2f2f2\" stroke-width=\"2\">\n";
  out << "<polygon points=\""
      << ego.x << ',' << ego.y - 12 << ' '
      << ego.x - 7 << ',' << ego.y + 9 << ' '
      << ego.x + 7 << ',' << ego.y + 9
      << "\" fill=\"#f2f2f2\"/>\n";
  out << "</g>\n";
}

void write_detections(
    std::ostream& out,
    const std::vector<Detection>& detections,
    const BevVisualizationConfig& config,
    bool use_label_palette,
    const Color& fixed_color) {
  int drawn = 0;
  for (const Detection& det : detections) {
    if (config.min_score >= 0.0f && det.score < config.min_score) {
      continue;
    }
    if (config.max_draw >= 0 && drawn >= config.max_draw) break;

    const int color_key = config.color_by_track_id ? det.query_index : det.label;
    const Color& color = use_label_palette
        ? kPalette[palette_index(color_key)]
        : fixed_color;
    const std::vector<PixelPoint> corners = box_corners(det, config);
    const PixelPoint center = world_to_pixel(det.x, det.y, config);
    const PixelPoint vel = world_to_pixel(det.x + det.vx, det.y + det.vy, config);

    out << "<g stroke=\"" << color_string(color)
        << "\" fill=\"" << color_string(color)
        << "\" fill-opacity=\"0.16\" stroke-width=\"2\">\n";
    out << "<polygon points=\"";
    write_polygon_points(out, corners);
    out << "\"/>\n";
    out << "<line x1=\"" << center.x << "\" y1=\"" << center.y
        << "\" x2=\"" << vel.x << "\" y2=\"" << vel.y
        << "\" stroke-opacity=\"0.65\"/>\n";
    out << "<text x=\"" << center.x + 4 << "\" y=\"" << center.y - 4
        << "\" font-size=\"12\" fill=\"" << color_string(color)
        << "\" stroke=\"none\">";
    if (config.show_track_id) {
      out << "id " << det.query_index << " c" << det.label << " ";
    } else {
      out << det.label << " ";
    }
    out << std::fixed << std::setprecision(2) << det.score << "</text>\n";
    out << "</g>\n";
    ++drawn;
  }
}

void write_panel(
    std::ostream& out,
    const std::vector<Detection>& detections,
    const BevVisualizationConfig& config,
    bool use_label_palette,
    const Color& fixed_color) {
  write_grid(out, config);
  write_ego(out, config);
  write_detections(out, detections, config, use_label_palette, fixed_color);
}

}  // namespace

void write_bev_svg(
    const std::string& path,
    const std::vector<Detection>& detections,
    const BevVisualizationConfig& config) {
  std::ofstream out(path);
  require(static_cast<bool>(out), "Failed to open BEV SVG output: " + path);

  out << "<svg xmlns=\"http://www.w3.org/2000/svg\" width=\""
      << config.image_width << "\" height=\"" << config.image_height
      << "\" viewBox=\"0 0 " << config.image_width << ' '
      << config.image_height << "\">\n";
  out << "<rect width=\"100%\" height=\"100%\" fill=\"#101418\"/>\n";
  write_panel(out, detections, config, true, Color{255, 94, 94});
  out << "</svg>\n";
  require(static_cast<bool>(out), "Failed to write BEV SVG output: " + path);
}

void write_bev_comparison_svg(
    const std::string& path,
    const std::vector<Detection>& gt_detections,
    const std::vector<Detection>& pred_detections,
    const BevVisualizationConfig& config) {
  std::ofstream out(path);
  require(static_cast<bool>(out), "Failed to open BEV SVG output: " + path);

  const int gap = 36;
  const int title_height = 52;
  const int panel_width = config.image_width;
  const int panel_height = config.image_height;
  const int width = panel_width * 2 + gap;
  const int height = panel_height + title_height;

  out << "<svg xmlns=\"http://www.w3.org/2000/svg\" width=\""
      << width << "\" height=\"" << height
      << "\" viewBox=\"0 0 " << width << ' ' << height << "\">\n";
  out << "<rect width=\"100%\" height=\"100%\" fill=\"#101418\"/>\n";
  out << "<text x=\"" << panel_width / 2 << "\" y=\"34\" "
      << "text-anchor=\"middle\" font-size=\"24\" fill=\"#f2f2f2\">GT boxes ("
      << gt_detections.size() << ")</text>\n";
  out << "<text x=\"" << panel_width + gap + panel_width / 2
      << "\" y=\"34\" text-anchor=\"middle\" font-size=\"24\" "
      << "fill=\"#f2f2f2\">TRT prediction (" << pred_detections.size()
      << ")</text>\n";

  out << "<g transform=\"translate(0," << title_height << ")\">\n";
  BevVisualizationConfig gt_config = config;
  gt_config.show_track_id = false;
  gt_config.color_by_track_id = false;
  write_panel(out, gt_detections, gt_config, false, Color{96, 210, 140});
  out << "</g>\n";

  out << "<g transform=\"translate(" << panel_width + gap << ','
      << title_height << ")\">\n";
  write_panel(out, pred_detections, config, true, Color{255, 94, 94});
  out << "</g>\n";
  out << "</svg>\n";
  require(static_cast<bool>(out), "Failed to write BEV SVG output: " + path);
}

}  // namespace uniad_lidar
