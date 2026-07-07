// GPU 3D voxelization — see gpu_voxelize.hpp.
#include "gpu_voxelize.hpp"

#include <cuda_fp16.h>
#include <cstdio>

namespace uniad_lidar {

namespace {

#define CUDA_OK(call)                                                       \
  do {                                                                      \
    cudaError_t _e = (call);                                                \
    if (_e != cudaSuccess) {                                                \
      std::fprintf(stderr, "[gpu_voxelize] CUDA error %s at %s:%d\n",       \
                   cudaGetErrorString(_e), __FILE__, __LINE__);             \
      return _e;                                                            \
    }                                                                       \
  } while (0)

// Kernel 1: one thread per point. Compute 3D voxel cell, atomicAdd the per-cell
// count, and atomicAdd the point features into the per-cell feature sum (only
// for the first max_points_per_voxel points, matching the CPU version).
__global__ void scatter_kernel(const float* points, size_t num_points, int fnum,
                               float xmin, float ymin, float zmin,
                               float xmax, float ymax, float zmax,
                               float vx, float vy, float vz,
                               int gx, int gy, int gz, int max_pts,
                               int* count, float* featsum) {
  size_t i = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (i >= num_points) return;
  const float px = points[i * fnum + 0];
  const float py = points[i * fnum + 1];
  const float pz = points[i * fnum + 2];
  if (px < xmin || px >= xmax || py < ymin || py >= ymax ||
      pz < zmin || pz >= zmax)
    return;
  const int cx = static_cast<int>(floorf((px - xmin) / vx));
  const int cy = static_cast<int>(floorf((py - ymin) / vy));
  const int cz = static_cast<int>(floorf((pz - zmin) / vz));
  if (cx < 0 || cx >= gx || cy < 0 || cy >= gy || cz < 0 || cz >= gz) return;
  const long cell = (static_cast<long>(cz) * gy + cy) * gx + cx;
  const int slot = atomicAdd(&count[cell], 1);
  if (slot >= max_pts) return;  // cap points per voxel (CPU parity)
  float* fs = featsum + cell * fnum;
  for (int f = 0; f < fnum; ++f) atomicAdd(&fs[f], points[i * fnum + f]);
}

// Kernel 2: one thread per grid cell. For non-empty cells, atomically claim a
// compact voxel id, write mean feature (fp16) + coors {0,cz,cy,cx}.
__global__ void compact_kernel(const int* count, const float* featsum, int fnum,
                               int gx, int gy, int gz, int max_pts,
                               int max_voxels, unsigned int* voxel_num,
                               __half* features, int* coors) {
  long cell = static_cast<long>(blockIdx.x) * blockDim.x + threadIdx.x;
  const long total = static_cast<long>(gz) * gy * gx;
  if (cell >= total) return;
  int c = count[cell];
  if (c <= 0) return;
  const int n = c < max_pts ? c : max_pts;  // divisor = capped count
  const unsigned int vid = atomicAdd(voxel_num, 1u);
  if (vid >= static_cast<unsigned int>(max_voxels)) return;
  const float* fs = featsum + cell * fnum;
  for (int f = 0; f < fnum; ++f)
    features[vid * fnum + f] = __float2half(fs[f] / static_cast<float>(n));
  const int cx = static_cast<int>(cell % gx);
  const int cy = static_cast<int>((cell / gx) % gy);
  const int cz = static_cast<int>(cell / (static_cast<long>(gx) * gy));
  coors[vid * 4 + 0] = 0;
  coors[vid * 4 + 1] = cz;
  coors[vid * 4 + 2] = cy;
  coors[vid * 4 + 3] = cx;
}

}  // namespace

GpuVoxelizer::GpuVoxelizer(const GpuVoxelizeConfig& cfg) : cfg_(cfg) {
  grid_cells_ = static_cast<size_t>(cfg.grid[0]) * cfg.grid[1] * cfg.grid[2];
  cudaMalloc(&d_count_, grid_cells_ * sizeof(int));
  cudaMalloc(&d_featsum_, grid_cells_ * cfg.feature_num * sizeof(float));
  cudaMalloc(&d_voxel_num_, sizeof(unsigned int));
  cudaMalloc(&d_features_,
             static_cast<size_t>(cfg.max_voxels) * cfg.feature_num *
                 sizeof(uint16_t));
  cudaMalloc(&d_coors_, static_cast<size_t>(cfg.max_voxels) * 4 * sizeof(int32_t));
}

GpuVoxelizer::~GpuVoxelizer() {
  cudaFree(d_count_);
  cudaFree(d_featsum_);
  cudaFree(d_voxel_num_);
  cudaFree(d_features_);
  cudaFree(d_coors_);
  if (d_points_) cudaFree(d_points_);
}

cudaError_t GpuVoxelizer::voxelize(const float* points_host, size_t num_points,
                                   cudaStream_t stream, void** out_features,
                                   void** out_coors, size_t* out_num_voxels) {
  const int fnum = cfg_.feature_num;
  // (re)alloc point staging buffer
  if (num_points > points_cap_) {
    if (d_points_) cudaFree(d_points_);
    CUDA_OK(cudaMalloc(&d_points_, num_points * fnum * sizeof(float)));
    points_cap_ = num_points;
  }
  CUDA_OK(cudaMemcpyAsync(d_points_, points_host, num_points * fnum * sizeof(float),
                          cudaMemcpyHostToDevice, stream));
  // clear dense buffers + compact counter
  CUDA_OK(cudaMemsetAsync(d_count_, 0, grid_cells_ * sizeof(int), stream));
  CUDA_OK(cudaMemsetAsync(d_featsum_, 0,
                          grid_cells_ * fnum * sizeof(float), stream));
  CUDA_OK(cudaMemsetAsync(d_voxel_num_, 0, sizeof(unsigned int), stream));

  const int gx = cfg_.grid[0], gy = cfg_.grid[1], gz = cfg_.grid[2];
  {
    dim3 threads(256);
    dim3 blocks(static_cast<unsigned int>((num_points + 255) / 256));
    scatter_kernel<<<blocks, threads, 0, stream>>>(
        d_points_, num_points, fnum,
        cfg_.range[0], cfg_.range[1], cfg_.range[2],
        cfg_.range[3], cfg_.range[4], cfg_.range[5],
        cfg_.voxel_size[0], cfg_.voxel_size[1], cfg_.voxel_size[2],
        gx, gy, gz, cfg_.max_points_per_voxel, d_count_, d_featsum_);
  }
  {
    dim3 threads(256);
    dim3 blocks(static_cast<unsigned int>((grid_cells_ + 255) / 256));
    compact_kernel<<<blocks, threads, 0, stream>>>(
        d_count_, d_featsum_, fnum, gx, gy, gz, cfg_.max_points_per_voxel,
        cfg_.max_voxels, d_voxel_num_, reinterpret_cast<__half*>(d_features_),
        d_coors_);
  }
  unsigned int nvox = 0;
  CUDA_OK(cudaMemcpyAsync(&nvox, d_voxel_num_, sizeof(unsigned int),
                          cudaMemcpyDeviceToHost, stream));
  CUDA_OK(cudaStreamSynchronize(stream));
  if (nvox > static_cast<unsigned int>(cfg_.max_voxels))
    nvox = cfg_.max_voxels;
  *out_features = d_features_;
  *out_coors = d_coors_;
  *out_num_voxels = nvox;
  return cudaSuccess;
}

}  // namespace uniad_lidar
