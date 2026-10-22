# Offline LightOp small-M W8A8 configuration patch

This opt-in tool adds 40 configuration entries to the **installed LightOp**
`config_smooth_gemm.py`. It does not modify SGLang dispatch, model weights,
quantization scales, or the LightOp fallback implementation. Nothing imports or
runs this tool during server startup. Changes affect every process using the
same LightOp installation, not just one SGLang checkout.

Validated target: **gfx936, 80 CUs**, LightOp
`0.6.0+dtk2604.torch2110.2610191203.ga289bd`. Other versions and unknown file
hashes are rejected. The flags below are operator confirmations, **not** automatic
hardware or process checks. Do not apply these configurations to another target.

## Use

Run from this directory with the Python interpreter used by the serving workers.
No extra Python dependencies, torch import, or GPU allocation are needed.

```bash
python3 config_patch.py check
```

Before applying or restoring, stop **all services sharing this installation**.
Do not patch an active precision/performance run. Review the target and then run:

```bash
python3 config_patch.py apply --yes --services-stopped --confirm-gfx936-80cu
```

The tool creates `config_smooth_gemm.py.sglang-small-m40.bak` beside the target,
checks the original and resulting SHA-256, and atomically replaces only that
configuration file. It refuses conflicting backups and concurrent tool runs.
Restart every affected worker to reload the configuration and clear cached
selection failures. No files are written by `check`.

To undo, stop the services again and run:

```bash
python3 config_patch.py restore --yes --services-stopped --confirm-gfx936-80cu
```

Restore accepts only the exact patched file and a verified original backup;
later edits are never silently overwritten. Backups are kept after restore.
An already manually patched installation is detected without changing it; if
this tool did not create a backup, automated restore is unavailable. Do not
fabricate one. Unknown versions/files need a separately reviewed patch.
If a process is killed mid-operation, a `.sglang-small-m40.lock` may remain.
Inspect the configuration hash and ensure no tool is running before manually
removing that exact lock. There is no force or automatic lock-stealing mode.

## Scope and evidence

`configs.json` is the sole entry list; values are
`[warps, nperblock, kperwarp, splitk, use_n_warps]`. The resulting file hash is
the same as the configuration used in the existing patched-LightOp validation.

* WKV `N=512, K=5120`: M=1..16. Service audit observed M=6,12 in eager execution
  and M=5,10,15 during graph capture/warmup. The other 11 are coverage additions.
* The other 24 entries match service graph-capture/warmup observations. Capture
  execution does not by itself prove a graph was replayed by a real request.
* All 40 have prior operator numerical validation, including graph replay;
  current service coverage does not establish every future shape or model.
* Model accuracy and throughput are separate measurements. Do not interpret
  microbenchmark speedups as end-to-end improvements or reuse historical model
  scores as results for a changed environment. Attach the completed evaluation
  reports to the PR, with source/dependency versions and simulation disabled.

## CPU tests

```bash
python3 -m unittest discover -s . -p 'test_config_patch.py' -v
```

Tests use a temporary installation fixture; they never patch the real package.
They validate installer behavior, not GPU numerical correctness. Reproducing
the pinned resulting SHA-256 against the original vendor file is a separate
artifact check; no full vendor configuration file is redistributed here.
