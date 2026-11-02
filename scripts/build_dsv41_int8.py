"""Build the optional, frozen auto-v1 HCU INT8 projection extension."""

import argparse
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--build-dir", type=Path, required=True)
    args = parser.parse_args()

    import torch
    from torch.utils.cpp_extension import ROCM_HOME, load

    if not torch.version.hip or ROCM_HOME is None:
        parser.error("A ROCm/DTK PyTorch installation and ROCM_HOME are required")

    # Keep the architecture boundary of the validated LightOp-derived kernel.
    # This optional extension is not required for the generic Triton backend.
    import os

    os.environ["PYTORCH_ROCM_ARCH"] = "gfx936"
    os.environ.setdefault("MAX_JOBS", "4")
    source = (
        Path(__file__).resolve().parents[1]
        / "python/sglang/srt/layers/quantization/csrc/dsv41_int8"
    )
    rocm = Path(ROCM_HOME)
    build = args.build_dir.resolve()
    build.mkdir(parents=True, exist_ok=True)
    library = load(
        name="wuju_dsv41_int8_complete",
        sources=[str(source / n) for n in ("int8_rowmajor.cpp", "lightop_small_m.cu")],
        extra_include_paths=[str(rocm / "include")],
        extra_cflags=["-O3", "-DUSE_ROCM"],
        extra_ldflags=[
            f"-L{rocm / 'lib'}",
            "-lrocblas",
            "-lc10_hip",
            "-ltorch_hip",
            f"-Wl,-rpath,{rocm / 'lib'}",
        ],
        build_directory=str(build),
        extra_cuda_cflags=["-O3", "-ffp-contract=off"],
        with_cuda=True,
        is_python_module=False,
        verbose=True,
    )
    print(library)


if __name__ == "__main__":
    main()
