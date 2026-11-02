// Copyright (c) 2026 Hygon Information Technology Co., Ltd.
// SPDX-License-Identifier: Apache-2.0
// Adapted from LightOp 5c866bec33c3385053c61b2869751c79ae8a4875,
// csrc/w8a8_smooth.cu, NWarps kernel. Changes: FP32 output, token-major
// groups in one launch, original scale order, no static workspace.
#include <ATen/ATen.h>
#include <c10/core/DeviceGuard.h>
#include <c10/hip/HIPStream.h>
#include <hip/hip_runtime.h>
#include <torch/library.h>
#include <type_traits>
namespace {
using intx4 = __attribute__((vector_size(16))) int;
using int8x8 = __attribute__((vector_size(8))) int8_t;
using int8x16 = __attribute__((vector_size(16))) int8_t;
using floatx4 = __attribute__((vector_size(16))) float;
using half4_t = __attribute__((vector_size(8))) _Float16;
using bf16x4_t = __attribute__((vector_size(8))) uint16_t;
template <int C> struct int8ABVector {
  union {
    int8x8 data[C];
    int8x16 data_16[C / 2];
  };
};
__device__ inline void builtin_amdgcn_mmac(const int8x8 &a, const int8x8 &b,
                                           intx4 &c) {
#if defined(__gfx936__)
  c = __builtin_amdgcn_mmac_i32_16x16x32i8(*reinterpret_cast<const long *>(&a),
                                           *reinterpret_cast<const long *>(&b),
                                           c);
#elif defined(__HIP_DEVICE_COMPILE__)
#error This isolated small-M kernel is validated only for gfx936
#endif
}
template <class T> __device__ inline T convert(float f) {
  if constexpr (std::is_same<T, float>::value)
    return f;
  else {
    uint32_t u = __float_as_uint(f);
    u += 0x7fff + ((u >> 16) & 1);
    return u >> 16;
  }
}
template <typename scalar_t, int NUM_WARPS = 4, int NPerBlock = 16,
          int KPerWarp = 64, bool UseBias = false>
__global__ __launch_bounds__(NUM_WARPS *
                             64) void small_int8(int8_t *a, int8_t *b,
                                                 scalar_t *d, float *scale_a,
                                                 float *scale_b, scalar_t *bias,
                                                 int M, int N, int K, int G) {
  const int group = blockIdx.y;
  a += group * K;
  b += group * N * K;
  d += group * N;
  scale_a += group;
  scale_b += group * N;
  constexpr int WarpK = KPerWarp;
  constexpr int tc_number = WarpK / 32;
  constexpr int k_per_thread = tc_number * 8;
  constexpr int NSizePerWarp = 16;
  constexpr int NLoop = NPerBlock / NSizePerWarp;
  const int tid = threadIdx.x;
  const int warp_idx = __builtin_amdgcn_readfirstlane(tid / 64);
  const int bid = blockIdx.x * NUM_WARPS + warp_idx;
  const int lane = tid % 64;
  const int row_id =
      lane %
      NSizePerWarp; // 不用改，warp内各个线程的layout与fp16的tensorcore一致，只是thread持有的元素数量变多了
  const int col_id = lane / NSizePerWarp;
  const bool row_id_valid = row_id < M;
  intx4 d_vec[NLoop];
  for (int i = 0; i < NLoop; i++) {
    d_vec[i] = {0, 0, 0, 0};
  }
  d += NPerBlock * bid;
  b += NPerBlock * bid * K;
  scale_b += NPerBlock * bid;

  const int N_lenth = N - NPerBlock * bid;

  using VecType = int8ABVector<tc_number>;
  int n_idx_valid[NLoop];
  for (int nn = 0; nn < NLoop; nn++) {
    int n_idx = NSizePerWarp * nn + row_id;
    if (n_idx < N_lenth) {
      n_idx_valid[nn] = n_idx * K;
    } else {
      n_idx_valid[nn] = -1;
    }
  }

  for (int i = 0; i < K; i += WarpK) {
    const int k_idx = i + col_id * k_per_thread;
    VecType a_vec{};
    if (row_id_valid) {
      a_vec = *(VecType *)(a + row_id * G * K + k_idx);
    }

    for (int nn = 0; nn < NLoop; nn++) {

      VecType b_vec{};
      if (n_idx_valid[nn] != -1) {
        b_vec = *(VecType *)(b + n_idx_valid[nn] + k_idx);
      }
#pragma unroll
      for (int ii = 0; ii < tc_number; ++ii) {
        builtin_amdgcn_mmac(a_vec.data[ii], b_vec.data[ii], d_vec[nn]);
      }
    }
  }

  extern __shared__ int out_smem_int[];
  int *out_smem_ptr =
      reinterpret_cast<int *>(out_smem_int + warp_idx * NPerBlock * M);

  using out_vec_dtype = typename std::conditional<
      std::is_same<scalar_t, float>::value, floatx4,
      typename std::conditional<std::is_same<scalar_t, _Float16>::value,
                                half4_t, bf16x4_t>::type>::type;
  if (row_id_valid) {
#pragma unroll
    for (int nn = 0; nn < NLoop; nn++) {
      int out_smem_offset = nn * NSizePerWarp * M;
#pragma unroll
      for (int i = 0; i < 4; i++) {
        out_smem_ptr[out_smem_offset + row_id * NSizePerWarp + i * 4 + col_id] =
            d_vec[nn][i];
      }
    }
  }
  __syncthreads();

  if (row_id_valid) {
    const float scale_a_value = scale_a[row_id * G];
    for (int nn = 0; nn < NLoop; nn++) {
      int out_smem_offset = nn * NSizePerWarp * M;

      bool n_idx_valid = nn * NSizePerWarp + col_id * 4 < N_lenth;
      if (n_idx_valid) {

        out_vec_dtype out_vec_16bit;
        intx4 out_vec_int =
            *reinterpret_cast<intx4 *>(out_smem_ptr + out_smem_offset +
                                       row_id * NSizePerWarp + col_id * 4);
        floatx4 scale_b_vec = *reinterpret_cast<floatx4 *>(
            scale_b + nn * NSizePerWarp + col_id * 4);
#pragma unroll
        for (int i = 0; i < 4; i++) {
          // Preserve the validated arithmetic order: float(acc)*xs*ws.
          out_vec_16bit[i] =
              convert<scalar_t>(static_cast<float>(out_vec_int[i]) *
                                scale_a_value * scale_b_vec[i]);
        }
        *reinterpret_cast<out_vec_dtype *>(d + row_id * G * N +
                                           nn * NSizePerWarp + col_id * 4) =
            out_vec_16bit;
      }
    }
  }
}

template <class T, int W, int N>
void launch(const at::Tensor &q, const at::Tensor &w, const at::Tensor &xs,
            const at::Tensor &ws, at::Tensor y, int m, int n, int k, int g,
            hipStream_t stream) {
  const int blocks = (n + W * N - 1) / (W * N);
  const int shared = W * N * m * sizeof(int32_t);
  small_int8<T, W, N, 256, false><<<dim3(blocks, g), W * 64, shared, stream>>>(
      const_cast<int8_t *>(q.data_ptr<int8_t>()),
      const_cast<int8_t *>(w.data_ptr<int8_t>()),
      reinterpret_cast<T *>(y.data_ptr()),
      const_cast<float *>(xs.data_ptr<float>()),
      const_cast<float *>(ws.data_ptr<float>()), nullptr, m, n, k, g);
}
template <class T>
void dispatch(const at::Tensor &q, const at::Tensor &w, const at::Tensor &xs,
              const at::Tensor &ws, at::Tensor y, int m, int n, int k, int g,
              int warps, int tile, hipStream_t stream) {
  if (warps == 4 && tile == 16)
    launch<T, 4, 16>(q, w, xs, ws, y, m, n, k, g, stream);
  else if (warps == 4 && tile == 32)
    launch<T, 4, 32>(q, w, xs, ws, y, m, n, k, g, stream);
  else if (warps == 4 && tile == 64)
    launch<T, 4, 64>(q, w, xs, ws, y, m, n, k, g, stream);
  else if (warps == 8 && tile == 16)
    launch<T, 8, 16>(q, w, xs, ws, y, m, n, k, g, stream);
  else if (warps == 8 && tile == 32)
    launch<T, 8, 32>(q, w, xs, ws, y, m, n, k, g, stream);
  else
    TORCH_CHECK(false, "Unsupported isolated LightOp launch configuration");
}
void smooth_out(const at::Tensor &q, const at::Tensor &w, const at::Tensor &xs,
                const at::Tensor &ws, at::Tensor y, int64_t warps,
                int64_t tile) {
  TORCH_CHECK(q.is_cuda() && w.is_cuda() && xs.is_cuda() && ws.is_cuda() &&
                  y.is_cuda(),
              "HCU tensors required");
  TORCH_CHECK(q.device() == w.device() && q.device() == xs.device() &&
                  q.device() == ws.device() && q.device() == y.device(),
              "All INT8 projection tensors must share one device");
  TORCH_CHECK(q.scalar_type() == at::kChar && w.scalar_type() == at::kChar &&
                  xs.scalar_type() == at::kFloat &&
                  ws.scalar_type() == at::kFloat,
              "Need INT8 weights/activations and FP32 scales");
  TORCH_CHECK(q.dim() == 3 && w.dim() == 2 && xs.dim() == 3 && ws.dim() == 2 &&
                  y.dim() == 3 && q.is_contiguous() && w.is_contiguous() &&
                  xs.is_contiguous() && ws.is_contiguous() && y.is_contiguous(),
              "Invalid small-M tensor layout");
  const auto m = q.size(0), g = q.size(1), k = q.size(2);
  TORCH_CHECK(m >= 1 && m <= 16 && g >= 1 && w.size(0) % g == 0 &&
                  w.size(1) == k,
              "Small-M INT8 requires 1<=M<=16 and matching grouped dimensions");
  const auto n = w.size(0) / g;
  TORCH_CHECK(k > 0 && k % 256 == 0 && n > 0 && n % 16 == 0 &&
                  k * g <= INT32_MAX && n * g <= INT32_MAX &&
                  n * k * g <= INT32_MAX,
              "Unsupported small-M dimensions");
  TORCH_CHECK(xs.size(0) == m && xs.size(1) == g && xs.size(2) == 1 &&
                  ws.size(0) == g * n && ws.size(1) == 1 && y.size(0) == m &&
                  y.size(1) == g && y.size(2) == n,
              "Invalid small-M output or channel scale shape");
  c10::DeviceGuard guard(q.device());
  auto stream = c10::hip::getCurrentHIPStream(q.get_device()).stream();
  if (y.scalar_type() == at::kFloat)
    dispatch<float>(q, w, xs, ws, y, m, n, k, g, warps, tile, stream);
  else if (y.scalar_type() == at::kBFloat16)
    dispatch<uint16_t>(q, w, xs, ws, y, m, n, k, g, warps, tile, stream);
  else
    TORCH_CHECK(false, "Small-M native output must be FP32 or BF16; use the "
                       "original INT8 path for FP16");
  TORCH_CHECK(hipGetLastError() == hipSuccess,
              "Isolated LightOp INT8 launch failed");
}
} // namespace
TORCH_LIBRARY_FRAGMENT(wuju_dsv41_i8, m) {
  m.def("smooth_out(Tensor q, Tensor w, Tensor xs, Tensor ws, Tensor(a!) y, "
        "int warps, int tile) -> ()");
}
TORCH_LIBRARY_IMPL(wuju_dsv41_i8, CUDA, m) { m.impl("smooth_out", smooth_out); }
