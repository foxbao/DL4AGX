// GPU 3D voxelization for the LiDAR sparse encoder front-end.
//
// Replaces the CPU unordered_map hashing + mean-VFE in voxelize_raw_points
// (~45 ms/frame, the pipeline bottleneck) with an on-GPU dense-grid +
// atomicAdd voxelizer that produces the SAME output libspconv expects:
//   features_half : [num_voxels, 4] fp16, per-voxel mean of point features
//   coors         : [num_voxels, 4] int32, {0, cz, cy, cx}
//
// Mechanism borrowed from NVIDIA CUDA-PointPillars (dense mask + atomicAdd),
// adapted to 3D voxels + mean-VFE (PointPillars is 2D pillars with a different
// 10-dim feature; here we replicate our simple 4-dim mean VFE exactly).
#ifndef UNIAD_SPARSE_LIDAR_GPU_VOXELIZE_HPP
#define UNIAD_SPARSE_LIDAR_GPU_VOXELIZE_HPP

#include <cstddef>
#include <cstdint>
#include <cuda_runtime.h>

namespace uniad_lidar {

struct GpuVoxelizeConfig {
  float range[6];       // xmin,ymin,zmin,xmax,ymax,zmax
  float voxel_size[3];  // vx,vy,vz
  int grid[3];          // gx,gy,gz
  int max_voxels;
  int max_points_per_voxel;
  int feature_num;      // 4
};

// Persistent device workspace (dense grid buffers), allocated once and reused.
class GpuVoxelizer {
 public:
  explicit GpuVoxelizer(const GpuVoxelizeConfig& cfg);
  ~GpuVoxelizer();

  // Voxelize points_host [num_points * feature_num] float32.
  // Outputs device pointers (owned by this object, valid until next call):
  //   *out_features : [*out_num_voxels, feature_num] fp16 (device)
  //   *out_coors    : [*out_num_voxels, 4] int32 (device)
  // Returns cudaError_t.
  cudaError_t voxelize(const float* points_host, size_t num_points,
                       cudaStream_t stream,
                       void** out_features, void** out_coors,
                       size_t* out_num_voxels);

 private:
  GpuVoxelizeConfig cfg_;
  size_t grid_cells_ = 0;
  // dense per-cell buffers
  int* d_count_ = nullptr;        // [grid_cells] point count per cell
  float* d_featsum_ = nullptr;    // [grid_cells * feature_num] feature sum
  // compact outputs
  unsigned int* d_voxel_num_ = nullptr;  // [1] atomic compact counter
  uint16_t* d_features_ = nullptr;        // [max_voxels * feature_num] fp16
  int32_t* d_coors_ = nullptr;            // [max_voxels * 4]
  float* d_points_ = nullptr;             // [max_points * feature_num]
  size_t points_cap_ = 0;
};

}  // namespace uniad_lidar

#endif  // UNIAD_SPARSE_LIDAR_GPU_VOXELIZE_HPP
