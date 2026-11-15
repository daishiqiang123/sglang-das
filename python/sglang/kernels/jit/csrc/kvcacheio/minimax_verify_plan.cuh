#pragma once

#include <sgl_kernel/utils.cuh>
#include <tvm/ffi/container/tensor.h>
#include <hipcub/block/block_scan.hpp>

namespace sglang {
namespace minimax_verify {

constexpr int Threads = 128;
constexpr int Count = 512;
constexpr int Items = Count / Threads;
constexpr int Page = 128;
constexpr int Width = 8;
constexpr int Heads = 4;
constexpr int Top = 16;
using Scan = hipcub::BlockScan<int, Threads>;

__global__ void plan_kernel(
    const int64_t* reqs, const int* real, const int* topk, const int* ends,
    const int64_t* host, const int64_t* device, int64_t* tags, int* hot,
    int* page_slots, int64_t* miss_src, int* miss_dst, int* miss_count,
    int64_t* verify_src, int* verify_dst, int* verify_count,
    int req_slots, int host_stride, int device_stride, int map_stride,
    int head_stride, int query_stride, int choice_stride) {
  const int row = blockIdx.x;
  const int tid = threadIdx.x;
  const int64_t request = reqs[row];
  if (row >= *real || request < 0 || request >= req_slots ||
      device[request * device_stride] <= 0) {
    for (int j = tid; j < Count; j += Threads) hot[row * Count + j] = -1;
    if (tid == 0) miss_count[row] = verify_count[row] = 0;
    return;
  }

  extern __shared__ int by_page[];
  int* owner = by_page;
  int* slot_by_page = by_page + map_stride;
  __shared__ Scan::TempStorage scan;
  __shared__ int occupied[Count];
  __shared__ int free_slots[Count];
  const int last = ends[row * Width + Width - 1];
  const int prefix = last - Width;
  const auto* device_row = device + request * device_stride;
  const auto* host_row = host + request * host_stride;
  auto* tags_row = tags + request * Count;
  auto* slot_row = page_slots + request * map_stride;

  for (int i = tid; i < map_stride; i += Threads) owner[i] = Count;
  for (int i = tid; i < Count; i += Threads) occupied[i] = 0;
  if (tid == 0) {
    slot_by_page[prefix / Page] = -1;
    slot_by_page[(last - 1) / Page] = -1;
  }
  __syncthreads();

  int pages[Items];
  for (int i = 0; i < Items; ++i) {
    const int j = tid * Items + i;
    const int q = j / (Heads * Top);
    const int h = j / Top % Heads;
    const int logical = topk[h * head_stride + (row * Width + q) * query_stride + (j % Top) * choice_stride];
    const bool valid = logical >= 0 && logical < map_stride &&
                       logical * Page < last && logical * Page < host_stride;
    pages[i] = valid ? logical : -1;
    if (valid) atomicMin(owner + logical, j);
  }
  __syncthreads();

  bool first[Items];
  int prior[Items];
  int needs_slot[Items];
  for (int i = 0; i < Items; ++i) {
    const int j = tid * Items + i;
    const int page = pages[i];
    first[i] = page >= 0 && owner[page] == j;
    prior[i] = first[i] ? slot_row[page] : -1;
    if (prior[i] < 0 || prior[i] >= Count || tags_row[prior[i]] != page) prior[i] = -1;
    if (prior[i] >= 0) occupied[prior[i]] = 1;
    needs_slot[i] = first[i] && prior[i] < 0;
  }
  __syncthreads();

  int free[Items];
  int free_at[Items];
  for (int i = 0; i < Items; ++i) free[i] = occupied[tid * Items + i] == 0;
  Scan(scan).ExclusiveSum(free, free_at);
  for (int i = 0; i < Items; ++i)
    if (free[i]) free_slots[free_at[i]] = tid * Items + i;
  __syncthreads();
  int need_at[Items];
  Scan(scan).ExclusiveSum(needs_slot, need_at);
  __syncthreads();

  int slot[Items];
  int miss[Items];
  int64_t source[Items];
  int destination[Items];
  bool reusable[Items];
  for (int i = 0; i < Items; ++i) {
    const int page = pages[i];
    slot[i] = prior[i] >= 0 ? prior[i] : needs_slot[i] ? free_slots[need_at[i]] : -1;
    source[i] = first[i] && page * Page < prefix ? host_row[page * Page] : -1;
    destination[i] = first[i] ? device_row[slot[i] * Page] / Page : 0;
    reusable[i] = first[i] && page < prefix / Page && destination[i] > 0 && source[i] >= 0;
    miss[i] = first[i] && source[i] >= 0 && destination[i] > 0 &&
              !(reusable[i] && tags_row[slot[i]] == page);
  }
  __syncthreads();

  int miss_at[Items];
  int total_misses;
  Scan(scan).ExclusiveSum(miss, miss_at, total_misses);
  for (int i = 0; i < Items; ++i) {
    if (!first[i]) continue;
    const int page = pages[i];
    slot_by_page[page] = destination[i];
    slot_row[page] = slot[i];
    tags_row[slot[i]] = reusable[i] ? page : -1;
    if (miss[i]) {
      miss_src[row * Count + miss_at[i]] = source[i] / Page;
      miss_dst[row * Count + miss_at[i]] = destination[i];
    }
  }
  __syncthreads();

  for (int i = 0; i < Items; ++i) {
    const int page = pages[i];
    const int out = page < 0 ? 0 : slot_by_page[page];
    hot[row * Count + tid * Items + i] = out > 0 ? out : -1;
  }
  if (tid == 0) {
    miss_count[row] = total_misses;
    int count = 0;
    for (int i = 0; i < Width; ++i) {
      const int pos = prefix + i;
      const int dst = slot_by_page[pos / Page];
      const int64_t src = device_row[Count * Page + i];
      if (dst <= 0 || src <= 0) continue;
      verify_src[row * Width + count] = src;
      verify_dst[row * Width + count] = dst * Page + pos % Page;
      ++count;
    }
    verify_count[row] = count;
  }
}

inline void plan(
    tvm::ffi::TensorView reqs, tvm::ffi::TensorView real, tvm::ffi::TensorView topk,
    tvm::ffi::TensorView ends, tvm::ffi::TensorView host_locs, tvm::ffi::TensorView device,
    tvm::ffi::TensorView tags, tvm::ffi::TensorView hot, tvm::ffi::TensorView page_slots,
    tvm::ffi::TensorView miss_src, tvm::ffi::TensorView miss_dst, tvm::ffi::TensorView miss_count,
    tvm::ffi::TensorView verify_src, tvm::ffi::TensorView verify_dst, tvm::ffi::TensorView verify_count) {
  const auto map_stride = page_slots.strides()[0];
  const auto gpu = host::LaunchKernel::resolve_device(reqs.device());
  host::LaunchKernel(reqs.size(0), Threads, gpu, 2 * map_stride * sizeof(int))(
      plan_kernel, static_cast<const int64_t*>(reqs.data_ptr()),
      static_cast<const int*>(real.data_ptr()), static_cast<const int*>(topk.data_ptr()),
      static_cast<const int*>(ends.data_ptr()), static_cast<const int64_t*>(host_locs.data_ptr()),
      static_cast<const int64_t*>(device.data_ptr()), static_cast<int64_t*>(tags.data_ptr()),
      static_cast<int*>(hot.data_ptr()), static_cast<int*>(page_slots.data_ptr()),
      static_cast<int64_t*>(miss_src.data_ptr()), static_cast<int*>(miss_dst.data_ptr()),
      static_cast<int*>(miss_count.data_ptr()), static_cast<int64_t*>(verify_src.data_ptr()),
      static_cast<int*>(verify_dst.data_ptr()), static_cast<int*>(verify_count.data_ptr()),
      static_cast<int>(host_locs.size(0)), static_cast<int>(host_locs.strides()[0]),
      static_cast<int>(device.strides()[0]), static_cast<int>(map_stride),
      static_cast<int>(topk.strides()[0]), static_cast<int>(topk.strides()[1]),
      static_cast<int>(topk.strides()[2]));
}

}  // namespace minimax_verify
}  // namespace sglang
