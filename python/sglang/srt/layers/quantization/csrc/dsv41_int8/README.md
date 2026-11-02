# DeepSeek V4.1 channelwise INT8 projections

This directory contains the optional native backend from the validated auto-v1
implementation. The runtime also supplies a Triton implementation; its default
backend is `reference`, which does not load or build a native library.

The supported checkpoint scheme is static symmetric INT8 weights, one FP32 scale
per output channel (`[N, 1]`), and dynamic symmetric per-token INT8 activations.
Weights stay in their checkpoint row-major layout. GEMM accumulates in INT32 and
scales in FP32. `--enable-fp32-lm-head` requests FP32 logits without an intermediate
BF16 rounding. Tied INT8 LM-head embeddings and LM-head LoRA are unsupported.

## Optional auto-v1 backend

On Linux with a matching HCU DTK/PyTorch toolchain, build from the repository root:

```bash
export ROCM_HOME=/opt/dtk
python scripts/build_dsv41_int8.py --build-dir /tmp/dsv41-int8-build
export SGLANG_DSV41_INT8_NATIVE_LIBRARY=/tmp/dsv41-int8-build/wuju_dsv41_int8_complete.so
export SGLANG_DSV41_INT8_PROJECTION_BACKEND=auto
export PYTHONPATH="$PWD/python${PYTHONPATH:+:$PYTHONPATH}"
```

The extension targets `gfx936`. `auto` selects measured configurations only on
`gfx936` with 80 compute units. It uses rocBLAS for full-vocabulary LM-head
projections with 32–128 rows and grouped wo_a projections with 64 or 128 rows;
other shapes use Triton. Other devices retain the reference Triton configuration.
An eligible native selection requires the built library; absence is an error,
not a silent backend substitution. Warm up each device/stream before HIP graph
capture. The library path is consumed during backend loading, not at compilation
of the Python module.

`reference` selects the original Triton configuration. `native` is an explicit
capability-testing mode: it can select the LightOp-derived small-M kernel, which
is not selected by `auto` because it was slower for the measured shapes.

`lightop_small_m.cu` is derived from LightOp commit
`5c866bec33c3385053c61b2869751c79ae8a4875`, `csrc/w8a8_smooth.cu`, under Apache-2.0.
It adds FP32 output and token-major grouped output while preserving the scale
multiplication order. The existing `wuju_dsv41_i8` operator namespace and extension
name are retained to keep the frozen implementation unchanged.

Activation quantization uses SGLang's `per_token_quant_int8`, preserving its
rounding and tiny-input rules. It deliberately does not substitute a LightOp
quantizer with different rounding behavior.

## CPU regression tests

Run in an environment with SGLang's HCU dependencies installed:

```bash
PYTHONPATH=python python -m unittest discover \
  -s test/registered/unit/models -p test_dsv41_channelwise_int8.py -v
```

The tests cover checkpoint selection, weight/scale validation, vocabulary scale
sharding, FP32 logits dispatch, Markov head whole/sharded/greedy paths, streaming
loading, and backend selection guards. CPU tests do not verify GPU arithmetic.
