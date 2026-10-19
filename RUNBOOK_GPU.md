# RUNBOOK_GPU — running the pipeline on the local GPU

Companion to `RUNBOOK.md`, which covers the CPU/Windows build. Everything in
that file still applies — the MadGraph build, the standalone trees, the
working-directory rule, the MinGW wrapper. This file covers only what changes
when the flow is evaluated on the GPU.

---

## 1. Hardware

| item | value |
|---|---|
| GPU | NVIDIA GeForce RTX 3060 **Laptop** GPU |
| VRAM | 6144 MiB total · ~4300 MiB free with a desktop session running |
| Driver | 546.92 (CUDA 12.3 runtime) |
| Disk headroom | ~172 GB free |

The ~1.6 GB held by the desktop is not available to us. Anything sized close to
6 GB will not fit; budget against **4.3 GB**, not 6.

---

## 2. A CUDA-enabled torch — **already installed**

The `workspace` environment originally carried `torch 2.3.0+cpu`, which is the
CPU-only build — `torch.cuda.is_available()` returns `False` and the engine
silently falls back to CPU. It has since been replaced. Current state, verified
on 2026-10-19:

| check | value |
|---|---|
| `torch.__version__` | `2.3.0+cu121` |
| `torch.version.cuda` | 12.1 |
| `torch.cuda.is_available()` | `True` |
| device | NVIDIA GeForce RTX 3060 Laptop GPU |
| capability | 8.6 |

If the environment is rebuilt, install the CUDA wheel explicitly — do not let a
plain `pip install torch` choose, since the PyPI Windows default is CPU-only:

```bash
/c/Users/laiji/.conda/envs/workspace/python.exe -m pip install \
    "torch==2.3.0+cu121" \
    --index-url https://download.pytorch.org/whl/cu121
```

**Why `+cu121` and not `+cu124`:** the installed driver (546.92) advertises CUDA
12.3. CUDA minor-version compatibility lets a 12.x runtime run under a 12.3
driver, so cu121 is the safe choice and cu124 is a needless risk.

**Why the same torch version:** pinning to 2.3.0 keeps the `numpy<2`
requirement from `RUNBOOK.md` §2.1 valid. Torch 2.3 is compiled against NumPy
1.x; changing the torch version would reopen that question.

**A second, unrelated trap:** the PATH default
(`C:\ProgramData\anaconda3\python.exe`) has `torch 2.3.0+cpu` in its user
site-packages, and it does not even import (`shm.dll` fails to load). No
command in this repository uses it; always invoke `$PY` explicitly.

The download is several GB — the torch wheel plus the `nvidia-*` runtime wheels
(cuDNN, cuBLAS, cuSPARSE). Budget 10–30 minutes depending on the connection.

**To revert to CPU:**

```bash
/c/Users/laiji/.conda/envs/workspace/python.exe -m pip install \
    "torch==2.3.0+cpu" \
    --index-url https://download.pytorch.org/whl/cpu
```

---

## 3. Verify before running anything

```bash
PY=/c/Users/laiji/.conda/envs/workspace/python.exe
$PY -c "
import torch
print('torch          ', torch.__version__)
print('cuda available ', torch.cuda.is_available())
print('cuda build     ', torch.version.cuda)
print('device         ', torch.cuda.get_device_name(0) if torch.cuda.is_available() else '-')
free, total = torch.cuda.mem_get_info()
print('vram free/total: %.2f / %.2f GiB' % (free/2**30, total/2**30))
x = torch.randn(1024, 1024, device='cuda')
print('matmul device  ', (x @ x).device)
"
```

`cuda available` must be `True`, a device name must print, and `matmul device`
must be `cuda:0`. If availability is still `False` after the install, the torch
build did not take — check `$PY -m pip show torch`.

That check only proves a tensor can be created. The full proof is
`scripts/verify_torch_gpu.py`, which also allocates device memory, times a
4096² matmul on both devices, runs the trained conditional marginal on both and
requires float64 parity, and backpropagates through it. Last run on this
machine:

| check | result |
|---|---|
| matmul, 4096² float32 | 18.1× faster on GPU (379 ms → 21 ms) |
| conditional marginal, batch 65,536 | 3.6× faster on GPU (126 ms → 35 ms) |
| max \|x_cuda − x_cpu\| | 2.0e-15 |
| gradient device | `cuda:0` |

Its exit status is non-zero if any check fails, so it can gate a CI job.

---

## 4. Running smoke on the GPU

```bash
export REPO="C:/Users/laiji/test/conditional-nf-integration"
export PY="/c/Users/laiji/.conda/envs/workspace/python.exe"

# CWD must be inside the standalone dir (RUNBOOK.md §5)
cd "$REPO/generated/gg_ttbar_standalone"

PYTHONPATH="$REPO" $PY -u \
    experiments/run_madgraph_process_conditional_copula.py \
    --config "$REPO/configs/processes/gg_ttbar.json" \
    --speed-profile smoke \
    --device cuda \
    --output "$REPO/results/gg_ttbar_copula_smoke_gpu"
```

Then validate on GPU:

```bash
cd "$REPO/generated/gg_ttbar_standalone"

PYTHONPATH="$REPO" $PY -u \
    experiments/validate_madgraph_process_integral.py \
    --config "$REPO/configs/processes/gg_ttbar.json" \
    --checkpoint "$REPO/results/gg_ttbar_copula_smoke_gpu/madgraph_gg_ttbar_2_final_model.pt" \
    --device cuda \
    --output "$REPO/results/gg_ttbar_validation_smoke_gpu"
```

Repeat for `gg_tth` and `uu_ww_4l`, or use the sweep script (§7).

**Write GPU results to a separate output directory.** The CPU runs already own
`results/<proc>_copula_smoke/`; keeping `_gpu` suffixes lets the two be compared
rather than silently overwritten. This is the same discipline that made the
Sobol reproduction check meaningful.

---

## 5. What changes versus the CPU runbook

| | CPU (RUNBOOK.md) | GPU |
|---|---|---|
| `--device` | `cpu` (or omitted) | **`cuda`** — worth passing explicitly, since the default is decided at import time by `torch.cuda.is_available()` |
| `--no-compile-kernels` | **passed** (the kernel-compile path is CUDA-only) | **drop it** — on GPU the deterministic integrand kernel *can* compile, and that is the point |
| output dirs | `results/<proc>_copula_smoke` | `results/<proc>_copula_smoke_gpu` |
| expected wall-clock | minutes | **smoke: about the same; thorough: modestly faster — see §6** |

Kernel compilation needs a C++ toolchain. MSVC 2022 BuildTools is on this
machine's `PATH`, so it should work; if compilation fails, the run still
proceeds on the eager path, and `--no-compile-kernels` is the escape hatch.

---

## 6. What the GPU does and does not accelerate

**This is the most important section in this file.** It is a measurement, not a
prediction.

### Measured: the GPU buys nothing on the **smoke** profile

`gg_ttbar`, `--speed-profile smoke`, identical config and budget:

| | CPU | GPU |
|---|---:|---:|
| training wall clock | **303 s** | **304 s** |
| learned graph | 1 edge | 1 edge |
| final ESS | 0.8054 | 0.8074 |
| final Î | 2.632163e-08 | 2.647052e-08 |

**0.3 % — i.e. nothing.** The two models are statistically indistinguishable
(same graph, ESS within 0.25 %, Î within 0.6 %; the residual scatter is just a
different training trajectory, since the RNG stream differs by device).

**This is a smoke result and it does not generalise.** Smoke runs at KL batch
8 192, below the crossover where kernel launches amortise. Subsequent
measurements (`RESULTS.md` §6.2–6.3) put the picture right:

| measurement | result |
|---|---|
| smoke at KL batch 32 768 (thorough's value) | 388 s CPU vs 365 s GPU, **1.06×** |
| flow alone at 30 000 points | **1.41×** faster on GPU |
| flow alone at 100 000 points | **4.80×** faster on GPU |
| validation, `--gpu-chunk` 10k → 100k | **1.65–2.23×**, no code change |

The correct statement is: **the GPU is launch-bound below roughly 30 000 points
and wins above it; once the flow is accelerated, the CPU matrix element becomes
the end-to-end ceiling.** The thorough runs (§9) have since been completed and
confirm the models still reproduce the published numbers.

The Sobol column of the GPU validation is still **bit-identical** to
`results_public/gg_ttbar_validation` at all five energies, so moving the flow to
the GPU does not compromise the build-level check.

### Why — two compounding causes

**(a) The matrix element is the ceiling, and it is on the CPU.** It is evaluated
one event per Python call:

```python
for index, event in enumerate(momenta):
    values[index] = float(self.evaluate(p, self.alpha_s, -1))
```

The published `gg_ttbar` validation ran 500 000 evaluations in 5.34 s — about
**93 600 evaluations/second** — *with the flow already on CUDA*. The
matrix-element throughput measured on this box, on **one CPU core**, is
**96 600/second**. Those numbers match, which says the matrix element dominates
the wall clock whether or not the flow is on a GPU.

**(b) The driver forces a host↔device round trip every chunk.** From
`run_madgraph_process_conditional_copula.py`:

```python
unit_numpy = unit.detach().cpu().double().numpy()          # GPU → host
for start in range(0, len(unit_numpy), me_chunk):
    pieces.append(evaluator.log_integrand(...))             # CPU matrix element
return torch.as_tensor(np.concatenate(pieces), device=unit.device, ...)  # host → GPU
```

So each chunk of `--me-chunk` (default 512) samples on the GPU, copies to the
host, evaluates the matrix element on one CPU core, and copies back. Whatever
the GPU wins on the flow is spent crossing the bus, and then the CPU does the
part that actually costs time.

### What this means for the Alpaka port — now measured

The original text here said an Alpaka port "cannot deliver a ≥10× throughput
gate." That was wrong about the flow. Fusing the per-dimension spline chain
into one kernel launch removes the per-call overhead that kept the PyTorch
marginal flat at ~32 ms. One marginal per call, `float32`, both sides on the
GPU (`mango-alpaka`, `RESULTS.md` §2):

| N | forward | inverse |
|---:|---:|---:|
| 1 | 10.4× | **254×** |
| 1 000 | 7.1× | **176×** |
| 10 000 | 1.3× | **33×** |
| 50 000 | 0.6× | **12.7×** |

The inverse — the direction the sampler uses — clears 10× at every batch size
measured. The forward loses above ~10 000 points to the host→device copy of
`raw`; a device-pointer interface would remove that.

What remains true is the end-to-end caveat: ≥10× on the flow does not become
≥10× on total event-generation rate while the matrix element runs on one CPU
core. Three routes still move that number:

1. **Remove the round trip** — co-resident sampling and evaluation. Validation
   already gained 1.65–2.23× just by raising `--gpu-chunk`; the training
   driver still copies every chunk.
2. **Move the matrix element to the GPU** — a substantially harder problem than
   the flow, and not currently in the proposal's scope. This is where the real
   10× lives.
3. **Choose a workload where the flow dominates** — high multiplicity, where
   the matrix element is relatively cheaper and the phase space harder.

Route 2 needs a decision: if the gate is defined on total event-generation rate
and the matrix element is CPU-bound, no amount of Alpaka work on the flow alone
will pass it.

### Practical knobs

`--me-chunk` (default 512) is the batch size handed to the matrix element and
therefore the knob that matters most for the real bottleneck. `--gpu-chunk`
(default 10 000) controls the flow-side batch. Raising `--me-chunk` amortises
Python-loop overhead per call; it does not change the per-event cost.

---

## 7. Sweep script

`scripts/run_all_windows.sh` targets the CPU layout. For GPU runs, either pass
the flags by hand as in §4, or override the output profile:

```bash
NFI_PROFILE=smoke bash scripts/run_all_windows.sh train gg_ttbar
```

Note that the script does not currently pass `--device cuda` or a `_gpu` output
suffix; until it does, prefer the explicit commands in §4.

---

## 8. VRAM budget and out-of-memory

With ~4.3 GB usable:

| setting | smoke | thorough |
|---|---:|---:|
| `marginal_train_conditions` | 32 | **1024** |
| `final_ess_size` | 5 000 | **100 000** |
| `batch_size` (cap) | 2048 | larger |

Smoke fits comfortably. The thorough runs (§9) also completed on this card at
the recorded chunk sizes (`gpu_chunk = 10 000` in the validation configs), so
the earlier prediction that thorough would not fit was too pessimistic. They
are long, not impossible: 5–13 h per process.

If you hit `torch.cuda.OutOfMemoryError`: lower `--gpu-chunk` first (it is the
per-forward batch), then `--uniform-fraction`-adjacent batch settings. Setting
`PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True` also helps with
fragmentation on a 6 GB card.

---

## 9. Thorough runs — completed

This section was written as "later, not now"; the runs were done on
2026-10-17/18. Wall clocks: `gg_ttbar` **5 h 05 m**, `gg_tth` **12 h 26 m**,
`uu_ww_4l` **13 h 15 m**, all at `--speed-profile thorough --device cuda`, with
held-out validation per energy. Results are in
`results/<proc>_copula_thorough_gpu/` and
`results/<proc>_validation_thorough_gpu/`; the numbers are in `RESULTS.md`
§6.5. Notes for a rerun:

* Thorough is roughly 100× the KL budget of smoke. On this GPU that is
  **5–13 hours per process** (measured), and it holds the GPU — and the
  desktop — for the duration.
* The `_smoke` vs `_thorough` and `_cpu` vs `_gpu` distinctions should stay in
  the output directory names, because the whole point of these runs is
  comparison — against the CPU runs here and against `results_public/`.

---

## 10. Quick reference

```bash
export REPO="C:/Users/laiji/test/conditional-nf-integration"
export PY="/c/Users/laiji/.conda/envs/workspace/python.exe"

# 1. GPU present?
nvidia-smi --query-gpu=name,memory.total,memory.free --format=csv

# 2. torch sees it — and prove it is really executing there
$PY -c "import torch; print(torch.__version__, torch.cuda.is_available())"
$PY scripts/verify_torch_gpu.py

# 3. train (smoke, GPU) — note: NO --no-compile-kernels
cd "$REPO/generated/gg_ttbar_standalone"
PYTHONPATH="$REPO" $PY -u experiments/run_madgraph_process_conditional_copula.py \
    --config "$REPO/configs/processes/gg_ttbar.json" \
    --speed-profile smoke --device cuda \
    --output "$REPO/results/gg_ttbar_copula_smoke_gpu"

# 4. validate (GPU)
PYTHONPATH="$REPO" $PY -u experiments/validate_madgraph_process_integral.py \
    --config "$REPO/configs/processes/gg_ttbar.json" --device cuda \
    --checkpoint "$REPO/results/gg_ttbar_copula_smoke_gpu/madgraph_gg_ttbar_2_final_model.pt" \
    --output "$REPO/results/gg_ttbar_validation_smoke_gpu"
```
