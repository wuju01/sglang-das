"""Native INT8 GEMM bridge; preserves the validated projection arithmetic."""

import os
from functools import lru_cache

import torch
import triton
import triton.language as tl


@triton.jit
def _scale_int32(
    ACC,
    XS,
    WS,
    Y,
    T: tl.constexpr,
    G: tl.constexpr,
    R: tl.constexpr,
    GROUP_MAJOR: tl.constexpr,
    B: tl.constexpr,
):
    j = tl.program_id(0) * B + tl.arange(0, B)
    valid = j < T * G * R
    row = j // R
    group = row % G
    token = row // G
    col = j % R
    if GROUP_MAJOR:
        raw = (group * T + token) * R + col
    else:
        raw = j
    acc = tl.load(ACC + raw, valid, other=0).to(tl.float32)
    xs = tl.load(XS + row, valid, other=0)
    ws = tl.load(WS + group * R + col, valid, other=0)
    tl.store(Y + j, acc * xs * ws, valid)


@lru_cache(maxsize=4)
def load_native(path=None):
    path = path or os.environ.get("SGLANG_DSV41_INT8_NATIVE_LIBRARY")
    if not path:
        raise RuntimeError(
            "Prepare and specify the INT8 native extension before serving"
        )
    torch.ops.load_library(path)


def project(
    x,
    w,
    s,
    groups,
    rank,
    *,
    output_dtype=None,
    tile_n=0,
    group_major=False,
    small_m=False,
    warps=4,
    tile=16,
):
    from sglang.kernels.ops.quantization.int8_kernel import per_token_quant_int8

    if (
        x.ndim != 3
        or x.shape[1] != groups
        or w.shape != (groups * rank, x.shape[2])
        or s.shape != (groups * rank, 1)
        or w.dtype != torch.int8
        or s.dtype != torch.float32
        or not w.is_contiguous()
        or not s.is_contiguous()
        or x.device != w.device
        or x.device != s.device
    ):
        raise ValueError("Invalid row-major grouped INT8 projection")
    output_dtype = output_dtype or x.dtype
    if x.dtype not in (
        torch.bfloat16,
        torch.float16,
        torch.float32,
    ) or output_dtype not in (torch.bfloat16, torch.float16, torch.float32):
        raise ValueError("INT8 projections require floating activations/output")
    t = x.shape[0]
    y = torch.empty((t, groups, rank), device=x.device, dtype=output_dtype)
    if not t:
        return y
    xq, xs = per_token_quant_int8(x.contiguous())
    if small_m:
        torch.ops.wuju_dsv41_i8.smooth_out(xq, w, xs, s, y, warps, tile)
        return y
    if group_major:
        xq = xq.transpose(0, 1).contiguous()
    raw = torch.empty(
        (groups, t, rank) if group_major else (t, groups, rank),
        device=x.device,
        dtype=torch.int32,
    )
    torch.ops.wuju_dsv41_i8.gemm_out(xq, w, raw, groups, tile_n, group_major)
    _scale_int32[(triton.cdiv(y.numel(), 1024),)](
        raw,
        xs,
        s,
        y,
        t,
        groups,
        rank,
        group_major,
        1024,
        num_warps=4,
        enable_fp_fusion=False,
    )
    return y
