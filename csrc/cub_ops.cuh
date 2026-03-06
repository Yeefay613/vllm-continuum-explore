#pragma once

// Simple reduction functors to avoid relying on cub::Sum / cub::Max
// which changed across CUDA versions (e.g., CUDA 13).
namespace vllm {
namespace cub_ops {

template <typename T>
struct Sum {
  __device__ __forceinline__ T operator()(const T& a, const T& b) const {
    return a + b;
  }
};

template <typename T>
struct Max {
  __device__ __forceinline__ T operator()(const T& a, const T& b) const {
    return a > b ? a : b;
  }
};

}  // namespace cub_ops
}  // namespace vllm
