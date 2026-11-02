// SPDX-License-Identifier: Apache-2.0
// Row-major W8A8 -> INT32 bridge for the installed DTK rocBLAS kernels.
#include <ATen/ATen.h>
#include <c10/core/DeviceGuard.h>
#include <c10/hip/HIPStream.h>
#include <climits>
#include <cstdint>
#include <map>
#include <memory>
#include <rocblas/rocblas.h>
#include <torch/library.h>

namespace {
void check(rocblas_status status) {
  TORCH_CHECK(status == rocblas_status_success, "DSV41 INT8 rocBLAS status ",
              static_cast<int>(status));
}
struct Handle {
  rocblas_handle value = nullptr;
  Handle() { check(rocblas_create_handle(&value)); }
  ~Handle() {
    if (value)
      rocblas_destroy_handle(value);
  }
};
rocblas_handle handle_for(int device, hipStream_t stream) {
  thread_local std::map<std::pair<int, uintptr_t>, std::unique_ptr<Handle>>
      handles;
  const auto key = std::make_pair(device, reinterpret_cast<uintptr_t>(stream));
  auto it = handles.find(key);
  if (it == handles.end()) {
    hipStreamCaptureStatus status;
    TORCH_CHECK(hipStreamIsCapturing(stream, &status) == hipSuccess,
                "Cannot query HIP capture state");
    TORCH_CHECK(status == hipStreamCaptureStatusNone,
                "Warm up the DSV41 INT8 backend before graph capture");
    it = handles.emplace(key, std::make_unique<Handle>()).first;
  }
  check(rocblas_set_stream(it->second->value, stream));
  return it->second->value;
}
void gemm_out(const at::Tensor &q, const at::Tensor &w, at::Tensor out,
              int64_t groups, int64_t tile_n, bool group_major) {
  TORCH_CHECK(q.is_cuda() && w.is_cuda() && out.is_cuda(),
              "HCU tensors required");
  TORCH_CHECK(q.device() == w.device() && q.device() == out.device(),
              "INT8 GEMM tensors must share one device");
  TORCH_CHECK(q.scalar_type() == at::kChar && w.scalar_type() == at::kChar &&
                  out.scalar_type() == at::kInt,
              "Need INT8 activations/weights and INT32 output");
  TORCH_CHECK(
      q.dim() == 3 && w.dim() == 2 && out.dim() == 3 && q.is_contiguous() &&
          w.is_contiguous() && out.is_contiguous(),
      "Need contiguous 3D activations/output and row-major [G*N,K] weights");
  TORCH_CHECK(groups > 0 && tile_n >= 0, "Invalid groups or tile size");
  const auto m = q.size(group_major ? 1 : 0), k = q.size(2);
  TORCH_CHECK(q.size(group_major ? 0 : 1) == groups && w.size(1) == k &&
                  w.size(0) % groups == 0,
              "Invalid grouped matrix dimensions");
  const auto n = w.size(0) / groups;
  TORCH_CHECK(out.size(group_major ? 0 : 1) == groups &&
                  out.size(group_major ? 1 : 0) == m && out.size(2) == n,
              "Invalid INT32 output dimensions");
  TORCH_CHECK(m <= INT_MAX && n <= INT_MAX && k <= INT_MAX &&
                  groups <= INT_MAX && groups * k <= INT_MAX &&
                  groups * n <= INT_MAX,
              "rocBLAS dimensions exceed INT32 ABI");
  TORCH_CHECK(k > 0 && k % 4 == 0 && n > 0,
              "INT8 rocBLAS requires positive K divisible by 4");
  TORCH_CHECK(groups == 1 || tile_n == 0,
              "Vocabulary tiling is for dense projections");
  if (m == 0)
    return;
  c10::DeviceGuard guard(q.device());
  auto stream = c10::hip::getCurrentHIPStream(q.get_device()).stream();
  auto handle = handle_for(q.get_device(), stream);
  const int32_t alpha = 1, beta = 0;
  const auto *xp = q.data_ptr<int8_t>();
  const auto *wp = w.data_ptr<int8_t>();
  auto *yp = out.data_ptr<int32_t>();
  if (groups > 1) {
    const auto ldx = group_major ? k : groups * k;
    const auto ldy = group_major ? n : groups * n;
    const auto sx = group_major ? m * k : k;
    const auto sy = group_major ? m * n : n;
    check(rocblas_gemm_strided_batched_ex(
        handle, rocblas_operation_transpose, rocblas_operation_none, n, m, k,
        &alpha, wp, rocblas_datatype_i8_r, k, n * k, xp, rocblas_datatype_i8_r,
        ldx, sx, &beta, yp, rocblas_datatype_i32_r, ldy, sy, yp,
        rocblas_datatype_i32_r, ldy, sy, groups, rocblas_datatype_i32_r,
        rocblas_gemm_algo_standard, 0, 0));
  } else {
    const auto step = tile_n ? tile_n : n;
    for (int64_t start = 0; start < n; start += step) {
      const auto cols = std::min(step, n - start);
      check(rocblas_gemm_ex(
          handle, rocblas_operation_transpose, rocblas_operation_none, cols, m,
          k, &alpha, wp + start * k, rocblas_datatype_i8_r, k, xp,
          rocblas_datatype_i8_r, k, &beta, yp + start, rocblas_datatype_i32_r,
          n, yp + start, rocblas_datatype_i32_r, n, rocblas_datatype_i32_r,
          rocblas_gemm_algo_standard, 0, 0));
    }
  }
}
} // namespace
TORCH_LIBRARY(wuju_dsv41_i8, m) {
  m.def("gemm_out(Tensor q, Tensor w, Tensor(a!) out, int groups, int tile_n, "
        "bool group_major) -> ()");
}
TORCH_LIBRARY_IMPL(wuju_dsv41_i8, CUDA, m) { m.impl("gemm_out", gemm_out); }
