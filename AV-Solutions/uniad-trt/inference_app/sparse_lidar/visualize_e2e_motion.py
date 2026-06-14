#!/usr/bin/env python3
import argparse
import math
from pathlib import Path

import numpy as np


PALETTE = [
    (0, 114, 189),
    (217, 83, 25),
    (237, 177, 32),
    (126, 47, 142),
    (119, 172, 48),
    (77, 190, 238),
    (162, 20, 47),
    (51, 160, 44),
    (166, 86, 40),
    (255, 127, 0),
    (106, 61, 154),
    (31, 120, 180),
    (227, 26, 28),
]


class Detection:
    def __init__(self, values):
        self.x = float(values[0])
        self.y = float(values[1])
        self.z = float(values[2])
        self.length = float(values[3])
        self.width = float(values[4])
        self.height = float(values[5])
        self.yaw = float(values[6])
        self.vx = float(values[7])
        self.vy = float(values[8])
        self.label = int(values[9])
        self.score = float(values[10])
        self.query_index = int(values[11])


def parse_args():
    parser = argparse.ArgumentParser(
        description="Render UniAD LiDAR E2E motion trajectories to BEV SVG.")
    parser.add_argument("output_dir", help="uniad_lidar_e2e output directory.")
    parser.add_argument(
        "--gt-dir",
        default=None,
        help="Optional directory containing gt_detections_*.txt.")
    parser.add_argument("--num-frames", type=int, default=None)
    parser.add_argument("--image-width", type=int, default=900)
    parser.add_argument("--image-height", type=int, default=1200)
    parser.add_argument(
        "--xy-range",
        nargs=4,
        type=float,
        default=[-64.0, -48.0, 64.0, 48.0],
        metavar=("X_MIN", "Y_MIN", "X_MAX", "Y_MAX"))
    parser.add_argument(
        "--top-modes",
        type=int,
        default=6,
        help="Number of trajectory modes to draw per object.")
    parser.add_argument(
        "--min-score",
        type=float,
        default=0.0,
        help="Minimum detection score to draw.")
    return parser.parse_args()


def color_string(color):
    return f"rgb({color[0]},{color[1]},{color[2]})"


def palette(index):
    return PALETTE[index % len(PALETTE)]


def scale_for(args):
    x_min, y_min, x_max, y_max = args.xy_range
    sx = args.image_height / (x_max - x_min)
    sy = args.image_width / (y_max - y_min)
    return min(sx, sy)


def world_to_pixel(x, y, args):
    x_min, y_min, x_max, y_max = args.xy_range
    scale = scale_for(args)
    used_width = (y_max - y_min) * scale
    used_height = (x_max - x_min) * scale
    offset_x = (args.image_width - used_width) * 0.5
    offset_y = (args.image_height - used_height) * 0.5
    px = offset_x + (y - y_min) * scale
    py = offset_y + used_height - (x - x_min) * scale
    return px, py


def read_detections(path):
    detections = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            values = line.split()
            if len(values) != 12:
                raise ValueError(f"Malformed detection line in {path}: {line}")
            detections.append(Detection(values))
    return detections


def resolve_gt_path(gt_dir, frame):
    if gt_dir is None:
        return None
    root = Path(gt_dir)
    candidates = [
        root / f"gt_detections_{frame:06d}.txt",
        root / f"gt_detections_{frame}.txt",
        root / f"frame_{frame:06d}" / "gt_detections.txt",
    ]
    for path in candidates:
        if path.exists():
            return path
    return None


def box_corners(det, args):
    c = math.cos(det.yaw)
    s = math.sin(det.yaw)
    half_l = det.length * 0.5
    half_w = det.width * 0.5
    local = [
        (-half_l, -half_w),
        (half_l, -half_w),
        (half_l, half_w),
        (-half_l, half_w),
    ]
    corners = []
    for lx, ly in local:
        x = det.x + lx * c - ly * s
        y = det.y + lx * s + ly * c
        corners.append(world_to_pixel(x, y, args))
    return corners


def point_list(points):
    return " ".join(f"{x:.2f},{y:.2f}" for x, y in points)


def write_grid(out, args):
    ego = world_to_pixel(0.0, 0.0, args)
    scale = scale_for(args)
    out.write('<g stroke="#2d3741" stroke-width="1" fill="none">\n')
    for x in range(-60, 61, 10):
        a = world_to_pixel(float(x), args.xy_range[1], args)
        b = world_to_pixel(float(x), args.xy_range[3], args)
        out.write(
            f'<line x1="{a[0]:.2f}" y1="{a[1]:.2f}" '
            f'x2="{b[0]:.2f}" y2="{b[1]:.2f}"/>\n')
    for y in range(-40, 41, 10):
        a = world_to_pixel(args.xy_range[0], float(y), args)
        b = world_to_pixel(args.xy_range[2], float(y), args)
        out.write(
            f'<line x1="{a[0]:.2f}" y1="{a[1]:.2f}" '
            f'x2="{b[0]:.2f}" y2="{b[1]:.2f}"/>\n')
    for radius in range(15, 61, 15):
        out.write(
            f'<circle cx="{ego[0]:.2f}" cy="{ego[1]:.2f}" '
            f'r="{radius * scale:.2f}"/>\n')
    out.write("</g>\n")
    out.write(
        f'<polygon points="{ego[0]:.2f},{ego[1] - 12:.2f} '
        f'{ego[0] - 7:.2f},{ego[1] + 9:.2f} '
        f'{ego[0] + 7:.2f},{ego[1] + 9:.2f}" fill="#f2f2f2"/>\n')


def write_detections(out, detections, args, fixed_color=None, show_ids=True):
    for det in detections:
        if det.score < args.min_score:
            continue
        color = fixed_color if fixed_color else palette(det.query_index)
        color_css = color_string(color)
        center = world_to_pixel(det.x, det.y, args)
        vel = world_to_pixel(det.x + det.vx, det.y + det.vy, args)
        out.write(
            f'<g stroke="{color_css}" fill="{color_css}" '
            'fill-opacity="0.16" stroke-width="2">\n')
        out.write(f'<polygon points="{point_list(box_corners(det, args))}"/>\n')
        out.write(
            f'<line x1="{center[0]:.2f}" y1="{center[1]:.2f}" '
            f'x2="{vel[0]:.2f}" y2="{vel[1]:.2f}" '
            'stroke-opacity="0.65"/>\n')
        label = f"ID {det.query_index}" if show_ids else f"{det.label} {det.score:.2f}"
        out.write(
            f'<text x="{center[0] + 5:.2f}" y="{center[1] - 6:.2f}" '
            'font-family="Arial, sans-serif" font-size="22" font-weight="700" '
            f'fill="{color_css}" stroke="#101418" stroke-width="4" '
            f'paint-order="stroke fill">{label}</text>\n')
        out.write("</g>\n")


def write_motion(out, detections, traj, scores, args):
    if len(detections) == 0:
        return
    modes_to_draw = min(args.top_modes, traj.shape[1])
    for det_index, det in enumerate(detections):
        if det.score < args.min_score:
            continue
        order = np.argsort(scores[det_index])[::-1][:modes_to_draw]
        best_mode = int(order[0])
        color = color_string(palette(det.query_index))
        for mode in order[::-1]:
            xy = traj[det_index, mode, :, :2]
            points = [world_to_pixel(det.x, det.y, args)]
            points.extend(world_to_pixel(det.x + float(dx), det.y + float(dy), args)
                          for dx, dy in xy)
            is_best = int(mode) == best_mode
            width = 4 if is_best else 2
            opacity = 0.95 if is_best else 0.22
            dash = "" if is_best else ' stroke-dasharray="4 5"'
            out.write(
                f'<polyline points="{point_list(points)}" fill="none" '
                f'stroke="{color}" stroke-width="{width}" '
                f'stroke-opacity="{opacity}" stroke-linecap="round" '
                f'stroke-linejoin="round"{dash}/>\n')
            if is_best:
                end = points[-1]
                out.write(
                    f'<circle cx="{end[0]:.2f}" cy="{end[1]:.2f}" r="4" '
                f'fill="{color}" fill-opacity="1"/>\n')


def write_planning(out, sdc_traj, args):
    if sdc_traj is None:
        return
    points = [world_to_pixel(0.0, 0.0, args)]
    points.extend(world_to_pixel(float(x), float(y), args) for x, y in sdc_traj)
    out.write(
        f'<polyline points="{point_list(points)}" fill="none" '
        'stroke="#f7f0a8" stroke-width="6" stroke-opacity="0.95" '
        'stroke-linecap="round" stroke-linejoin="round"/>\n')
    for idx, point in enumerate(points[1:], start=1):
        radius = 5 if idx == len(points) - 1 else 3
        out.write(
            f'<circle cx="{point[0]:.2f}" cy="{point[1]:.2f}" '
            f'r="{radius}" fill="#f7f0a8" fill-opacity="1"/>\n')


def write_panel(out, detections, traj, scores, args, sdc_traj=None):
    write_grid(out, args)
    write_planning(out, sdc_traj, args)
    write_motion(out, detections, traj, scores, args)
    write_detections(out, detections, args)


def write_motion_svg(path, detections, traj, scores, args, sdc_traj=None):
    with open(path, "w") as out:
        out.write(
            f'<svg xmlns="http://www.w3.org/2000/svg" width="{args.image_width}" '
            f'height="{args.image_height}" viewBox="0 0 {args.image_width} '
            f'{args.image_height}">\n')
        out.write('<rect width="100%" height="100%" fill="#101418"/>\n')
        write_panel(out, detections, traj, scores, args, sdc_traj)
        out.write("</svg>\n")


def write_compare_svg(
        path, gt_detections, detections, traj, scores, args, sdc_traj=None):
    gap = 36
    title_height = 52
    width = args.image_width * 2 + gap
    height = args.image_height + title_height
    with open(path, "w") as out:
        out.write(
            f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" '
            f'height="{height}" viewBox="0 0 {width} {height}">\n')
        out.write('<rect width="100%" height="100%" fill="#101418"/>\n')
        out.write(
            f'<text x="{args.image_width / 2:.1f}" y="34" '
            'text-anchor="middle" font-size="24" fill="#f2f2f2">'
            f'GT boxes ({len(gt_detections)})</text>\n')
        out.write(
            f'<text x="{args.image_width + gap + args.image_width / 2:.1f}" '
            'y="34" text-anchor="middle" font-size="24" fill="#f2f2f2">'
            f'TRT prediction + motion ({len(detections)})</text>\n')
        out.write(f'<g transform="translate(0,{title_height})">\n')
        write_grid(out, args)
        write_detections(out, gt_detections, args, fixed_color=(96, 210, 140),
                         show_ids=False)
        out.write("</g>\n")
        out.write(
            f'<g transform="translate({args.image_width + gap},{title_height})">\n')
        write_panel(out, detections, traj, scores, args, sdc_traj)
        out.write("</g>\n</svg>\n")


def load_motion(output_dir, frame, num_dets):
    traj_path = output_dir / f"frame_{frame:06d}_traj.bin"
    scores_path = output_dir / f"frame_{frame:06d}_traj_scores.bin"
    traj = np.fromfile(traj_path, dtype=np.float32)
    scores = np.fromfile(scores_path, dtype=np.float32)
    expected_traj = num_dets * 6 * 12 * 5
    expected_scores = num_dets * 6
    if traj.size != expected_traj:
        raise ValueError(
            f"{traj_path} has {traj.size} floats, expected {expected_traj}.")
    if scores.size != expected_scores:
        raise ValueError(
            f"{scores_path} has {scores.size} floats, expected {expected_scores}.")
    return traj.reshape(num_dets, 6, 12, 5), scores.reshape(num_dets, 6)


def load_planning(output_dir, frame):
    path = output_dir / f"frame_{frame:06d}_sdc_traj.bin"
    if not path.exists():
        return None
    traj = np.fromfile(path, dtype=np.float32)
    if traj.size % 2 != 0:
        raise ValueError(f"{path} has {traj.size} floats, expected pairs.")
    return traj.reshape(-1, 2)


def frame_indices(output_dir, num_frames):
    if num_frames is not None:
        return range(num_frames)
    frames = []
    for path in sorted(output_dir.glob("frame_*_detections.txt")):
        stem = path.stem
        frames.append(int(stem.split("_")[1]))
    return frames


def main():
    args = parse_args()
    output_dir = Path(args.output_dir)
    for frame in frame_indices(output_dir, args.num_frames):
        detections = read_detections(output_dir / f"frame_{frame:06d}_detections.txt")
        traj, scores = load_motion(output_dir, frame, len(detections))
        sdc_traj = load_planning(output_dir, frame)
        write_motion_svg(
            output_dir / f"frame_{frame:06d}_motion.svg",
            detections,
            traj,
            scores,
            args,
            sdc_traj)
        gt_path = resolve_gt_path(args.gt_dir, frame)
        if gt_path is not None:
            write_compare_svg(
                output_dir / f"frame_{frame:06d}_motion_compare.svg",
                read_detections(gt_path),
                detections,
                traj,
                scores,
                args,
                sdc_traj)
    print(f"Wrote motion SVG frames to {output_dir}")


if __name__ == "__main__":
    main()
