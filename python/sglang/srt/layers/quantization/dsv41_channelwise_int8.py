"""Channelwise W8A8 special projections for DeepSeek V4.1 on HCU.

Checkpoint weights remain row-major INT8. The grouped contraction uses INT8
dot products, INT32 accumulation and FP32 activation/weight scale epilogues.
"""

import os
from functools import lru_cache

import torch
import triton
import triton.language as tl
from torch import nn

from sglang.srt.layers.quantization.base_config import QuantizeMethodBase


def enabled(layer, quant_config, prefix):
    if quant_config is None or quant_config.get_name() not in (
        "compressed_tensors",
        "compressed-tensors",
    ):
        return False
    scheme = quant_config.get_scheme_dict(layer, layer_name=prefix)
    if scheme is None:
        return False
    w, a = scheme.get("weights"), scheme.get("input_activations")
    if w is None or w.type != "int":
        return False
    if not (
        w.num_bits == 8
        and w.strategy == "channel"
        and w.symmetric
        and not w.dynamic
        and a is not None
        and a.num_bits == 8
        and a.type == "int"
        and a.strategy == "token"
        and a.symmetric
        and a.dynamic
        and (scheme.get("format") or quant_config.quant_format) == "int-quantized"
    ):
        raise ValueError(
            f"Unsupported INT8 projection scheme for {prefix}; need channel W8/token A8"
        )
    from sglang.srt.utils import is_hcu

    if not is_hcu():
        raise ValueError("DeepSeek V4.1 INT8 projection runtime currently requires HCU")
    return True


@triton.jit
def _grouped_int8_mm(
    X,
    W,
    XS,
    WS,
    Y,
    T: tl.constexpr,
    G: tl.constexpr,
    R: tl.constexpr,
    K: tl.constexpr,
    BM: tl.constexpr,
    BN: tl.constexpr,
    BK: tl.constexpr,
):
    tm = tl.program_id(0) * BM + tl.arange(0, BM)
    rn = tl.program_id(1) * BN + tl.arange(0, BN)
    group = tl.program_id(2)
    kk = tl.arange(0, BK)
    acc = tl.full((BM, BN), 0, tl.int32)
    for block in range(tl.cdiv(K, BK)):
        k = block * BK + kk
        x = tl.load(
            X + (tm[:, None] * G + group) * K + k[None, :],
            (tm[:, None] < T) & (k[None, :] < K),
            other=0,
        )
        w = tl.load(
            W + (group * R + rn[None, :]) * K + k[:, None],
            (rn[None, :] < R) & (k[:, None] < K),
            other=0,
        )
        acc += tl.dot(x, w)
    sx = tl.load(XS + tm * G + group, tm < T, other=0)
    sw = tl.load(WS + group * R + rn, rn < R, other=0)
    y = acc.to(tl.float32) * sx[:, None] * sw[None, :]
    tl.store(
        Y + (tm[:, None] * G + group) * R + rn[None, :],
        y,
        (tm[:, None] < T) & (rn[None, :] < R),
    )


@lru_cache(maxsize=None)
def _is_gfx936_80cu(device):
    props = torch.cuda.get_device_properties(device)
    return (
        getattr(props, "gcnArchName", "").split(":")[0] == "gfx936"
        and props.multi_processor_count == 80
    )


def _gemm_config(tokens, groups, rank, k, *, gfx936_80cu):
    """Measured decode configurations; preserve the generic INT8 path elsewhere.

    Larger N tiles reduce the number of programs in the decode N grid.
    Small-M long-K projections benefit from BK=256, while larger M would
    increase register pressure with that K tile. Neither choice changes the
    INT32 contraction or the FP32 scale multiplication order.
    """
    if (
        gfx936_80cu
        and 0 < tokens <= 96
        and (
            (groups, rank, k) == (8, 1024, 4096)
            or (groups, rank, k) == (1, 129280, 5120)
        )
    ):
        return (16, 128, 256 if tokens <= 16 else 128, 4)
    return (16, 64, 128, 4)


def _projection_plan(tokens, groups, rank, k, output_dtype, *, mode, gfx936_80cu):
    """Select only measured gfx936 paths; keep the original backend by default."""
    if mode not in ("reference", "auto", "native"):
        raise ValueError("INT8 projection backend must be reference, auto or native")
    reference = ("triton", 16, 64, 128, 4)
    if mode == "reference" or not gfx936_80cu or tokens <= 0:
        return reference
    if mode == "native":
        if (
            tokens <= 16
            and k % 256 == 0
            and rank % 16 == 0
            and output_dtype in (torch.bfloat16, torch.float32)
        ):
            return ("lightop", 4, 16)
        if k % 4 == 0:
            return ("rocblas",)
        return reference
    if 0 < tokens <= 128 and output_dtype in (torch.bfloat16, torch.float32):
        if groups == 1 and rank == 129280 and k == 5120 and tokens >= 32:
            return ("rocblas",)
        if (groups, rank, k) == (8, 1024, 4096) and tokens in (64, 128):
            return ("rocblas",)
    if 0 < tokens <= 128 and groups == 1 and rank in (129280, 16160) and k == 256:
        return ("triton", 16 if tokens <= 16 else 32, 128, 256, 4)
    return ("triton",) + _gemm_config(tokens, groups, rank, k, gfx936_80cu=gfx936_80cu)


def _grouped(x, w, s, groups, rank, *, output_dtype=None):
    from sglang.kernels.ops.quantization.int8_kernel import per_token_quant_int8

    if (
        x.ndim != 3
        or x.shape[1] != groups
        or w.dtype != torch.int8
        or w.shape != (groups * rank, x.shape[-1])
        or s.shape != (groups * rank, 1)
        or s.dtype != torch.float32
        or not w.is_contiguous()
        or not s.is_contiguous()
    ):
        raise ValueError("Invalid grouped INT8 wo_a weight, scale or activation layout")
    if x.dtype not in (torch.bfloat16, torch.float16, torch.float32):
        raise ValueError(
            "INT8 wo_a expects floating-point activations before quantization"
        )
    mode = os.environ.get("SGLANG_DSV41_INT8_PROJECTION_BACKEND", "reference")
    plan = _projection_plan(
        x.shape[0],
        groups,
        rank,
        x.shape[-1],
        output_dtype or x.dtype,
        mode=mode,
        gfx936_80cu=(mode != "reference" and x.is_cuda and _is_gfx936_80cu(x.device)),
    )
    if plan[0] in ("rocblas", "lightop"):
        from sglang.srt.layers.quantization import dsv41_int8_native as native

        native.load_native()
        return native.project(
            x,
            w,
            s,
            groups,
            rank,
            output_dtype=output_dtype,
            small_m=plan[0] == "lightop",
            warps=plan[1] if plan[0] == "lightop" else 4,
            tile=plan[2] if plan[0] == "lightop" else 16,
        )
    y = torch.empty(
        (x.shape[0], groups, rank), device=x.device, dtype=output_dtype or x.dtype
    )
    if y.numel() == 0:
        return y
    xq, xs = per_token_quant_int8(x.contiguous())
    bm, bn, bk, warps = plan[1:]
    _grouped_int8_mm[(triton.cdiv(x.shape[0], bm), triton.cdiv(rank, bn), groups)](
        xq,
        w,
        xs,
        s,
        y,
        x.shape[0],
        groups,
        rank,
        x.shape[-1],
        BM=bm,
        BN=bn,
        BK=bk,
        num_warps=warps,
        num_stages=2,
    )
    return y


def grouped(x, layer, groups, rank):
    return _grouped(x, layer.weight, layer.weight_scale, groups, rank)


def linear(x, weight, scale, *, output_dtype=None):
    """Run a row-major W8A8 projection without dequantizing the weight."""
    if x.ndim < 2 or weight.ndim != 2:
        raise ValueError("INT8 projection expects matrix weights and batched inputs")
    shape = x.shape
    y = _grouped(
        x.reshape(-1, 1, shape[-1]),
        weight,
        scale,
        1,
        weight.shape[0],
        output_dtype=output_dtype,
    )
    return y.view(*shape[:-1], weight.shape[0])


class ChannelwiseInt8Method(QuantizeMethodBase):
    supports_lm_head_lora = False

    def process_weights_after_loading(self, layer):
        for name in ("weight", "weight_scale"):
            if not getattr(getattr(layer, name), "_int8_projection_loaded", False):
                raise ValueError(f"Missing INT8 projection {name}")
        scale = layer.weight_scale
        valid_rows = getattr(layer, "_int8_projection_valid_rows", scale.shape[0])
        if not torch.all(torch.isfinite(scale)) or not torch.all(
            scale[:valid_rows] > 0
        ):
            raise ValueError("INT8 projection requires finite positive channel scales")

    def apply(self, layer, x, bias=None):
        return self.apply_with_output_dtype(layer, x, bias, output_dtype=x.dtype)

    def apply_with_output_dtype(self, layer, x, bias=None, *, output_dtype):
        y = linear(x, layer.weight, layer.weight_scale, output_dtype=output_dtype)
        return y if bias is None else y + bias


def install(layer):
    """Keep the original TP/padded-vocabulary loader for weights and row scales."""
    from sglang.srt.model_loader.weight_utils import default_weight_loader

    old = layer.weight
    loader = getattr(old, "weight_loader", default_weight_loader)
    output_dim = getattr(old, "output_dim", None)
    shard = getattr(layer, "shard_indices", None)
    layer._int8_projection_valid_rows = (
        shard.org_vocab_end_index - shard.org_vocab_start_index
        if shard is not None
        else old.shape[0]
    )
    for name, shape, dtype in (
        ("weight", old.shape, torch.int8),
        ("weight_scale", (old.shape[0], 1), torch.float32),
    ):
        param = nn.Parameter(
            torch.empty(shape, dtype=dtype, device=old.device), requires_grad=False
        )
        if output_dim is not None:
            param.output_dim = output_dim
        if name == "weight":
            param.input_dim = 1

        def load(param, value, *, _loader=loader, _name=name):
            if _name == "weight" and value.dtype != torch.int8:
                raise ValueError("INT8 projection requires INT8 checkpoint weights")
            if _name == "weight_scale" and (
                value.dtype != torch.float32 or value.ndim != 2 or value.shape[1] != 1
            ):
                raise ValueError("INT8 projection requires FP32 [N, 1] scales")
            _loader(param, value)
            param._int8_projection_loaded = True

        param.weight_loader = load
        setattr(layer, name, param)
    layer.quant_method = ChannelwiseInt8Method()
