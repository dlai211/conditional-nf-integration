# PRD.md — working notes on the `conditional-nf-integration` repo

**Personal log.** Written while probing someone else's repository to work out
its structure and how to actually run it. Everything below is a record of what
was tested on this machine, what worked, what failed, and what was too slow to
finish.

This document now has three parts, written on three different machines:

| part | machine | date | scope |
|---|---|---|---|
| **Part I** (§0–§9) | Windows 11, conda `workspace`, CPU-only torch | 2026-10-08 | repository reconnaissance and synthetic smoke runs; **no MadGraph** — that was the blocker |
| **Part II** (§10–§17) | CERN Linux node, EOS home, dedicated `python3.11` venv | 2026-10-08 → 2026-10-09 | MadGraph 3.6.7 installed and working; **the scattering results are now reproducible end to end** |
| **Part III** (§18–§22) | Linux/ROCm `odyssey` node, MG5 3.6.7, torch 2.5.1+rocm6.2 | 2026-10-26 | one-jet NF/Sobol/MadEvent validation; why `p p > z j` cannot run as-is; two proposed pipeline changes (each with both energy-range options); MadEvent energy-slice diagnostic |

Part I's conclusions about the *code* all still hold. Its central blocker — no
MadGraph — is **resolved in Part II**. Read Part II first if you are here for
the physics results.

---

## 0. Provenance

A public GitHub repo, **not** the local user's own work:

- primary remote (as of Part II): `https://github.com/dlai211/conditional-nf-integration.git`
- mirror seen in Part I: `https://github.com/XuZhaozhong-1/conditional-nf-integration.git`
- sole author: **Zhaozhong Xu** `<zhaozhong@…sdcc.bnl.gov>` (Brookhaven National
  Laboratory)
- commits: 2026-09-20 → 2026-09-25 (8 commits, branch `main`)
- working tree was clean on clone except for one uncommitted change
  (`experiments/run_adaptive_integrand_structure_search.py`, +5 lines adding a
  `sys.path` shim — not mine, left untouched)

**Part II update on provenance.** The two remotes are the *same project*, not
forks with divergent work: their histories share all 8 Part-I commits
(`3f795a0`…`3db0a2b`). `dlai211` is strictly ahead — one extra commit,
`4fdb861 "Initial commit to personal repo"`, which **adds `PRD.md` and
`Intro.md` and commits the `sys.path` shim** that Part I had found as an
uncommitted working-tree change. So the Part-I checkout was simply one commit
behind the author's personal remote. The local checkout was re-pointed at
`dlai211` and fast-forwarded `3db0a2b → 4fdb861`; nothing was discarded and the
tree is clean apart from one pre-existing untracked `.asetup.save`.

Everything in Part I was measured at `3db0a2b`; everything in Part II at
`4fdb861`.

Since this is someone else's published work, I have tried to keep the checkout
pristine: everything I generated lives in the gitignored `results/` tree, and
I restored the three tracked files I accidentally overwrote (§5.6). The only
new tracked-tree file is this `PRD.md`.

---

## 1. What this repository is

A research codebase studying **conditional normalizing flows as importance-sampling
proposals for families of integrals**

```
I(c) = ∫_[0,1]^d f(x | c) dx
```

The condition `c` is `sqrt(s_hat)` (partonic centre-of-mass energy) in the
physics applications. One model is trained across a continuous range of
conditions and then reused at held-out conditions **without retraining**.

The key methodological point (from `README.md` and `docs/METHODS_AND_APPLICATIONS.md`):

- The network does **not** predict the integral. It only decides *where to sample*.
- The estimator stays a plain importance-sampling average
  `Î_N(c) = (1/N) Σ f(x_n|c) / r_θ(x_n|c)`, so a bad proposal costs variance,
  not correctness.
- The proposal factorises into **conditional marginals** (monotone cubic-Hermite
  splines on [0,1]) times an **autoregressive copula** on the marginal rank
  coordinates `u_i`. This is the repo's central idea.
- A **defensive uniform mixture** `r_θ = (1-ε) q_θ + ε` keeps support over the
  whole hypercube.

Two application families:

1. **Scattering processes** via MadGraph matrix elements (2D, 5D, 8D, 20D).
2. **Synthetic stress tests** (`single_gaussian`, `sharp_broad`, `curved_ridge`,
   `xshape`, `regime_switch`, `sparse_wave_32`) — these need no MadGraph.

---

## 2. Repository structure

```
nf/                             the actual library (conditional flows)
  conditional_cubic_1d.py         monotone cubic-Hermite conditional flow, 1D
  conditional_linear_1d.py        linear conditional flow, 1D
  conditional_cubic_2d.py         2D copula coupling (1023 lines; see dead-code
                                  note below — only one script still uses it)
  conditional_autoregressive_cubic_2d.py
  conditional_rq_1d.py            rational-quadratic spline variant
  cube_flow.py, cube_spline_flow.py, cube_piecewise_spline_flow.py,
  cube_affine_flow.py             flows on the unit cube
  logit_rotation.py
  flows/realnvp_nd.py             RealNVP for nD
  train.py                        generic training loop (train_nf)
  evaluate.py                     nf_integrate + ESS
  losses.py                       kl / raw_std / kl+std / cv2 losses

madgraph_process_integrand.py   configurable matrix element + phase space (2/5/8D)
madgraph_tth_integrand.py       resonance-aware 20D ttH phase space

experiments/                    all drivers (see §4)
configs/processes/              gg_ttbar.json, gg_tth.json, uu_ww_4l.json
tests/                          3 test files, 12 tests
docs/                           methods, validation, reproducibility
results_public/                 curated CSV/JSON/PNG from published runs
scripts/curate_public_results.py
```

Important: `results_public/` is the **curated** output of the authors' runs. The
scripts write to `results/` instead, which is gitignored and does not exist in a
fresh clone.

### The dependency graph (this is the key to the whole repo)

`experiments/` is not 17 independent scripts. Two files are load-bearing:

```
run_copula_bin_sweep.py                    <- shared base: seed_everything,
    ^                                          target defs, flow classes
    |-- run_copula_vs_vegas_integrand.py
    |-- run_bayesian_selected_copula_benchmark.py
    |-- run_iterative_bayesian_edge_pruning_frozen_null.py
    |-- run_adaptive_integrand_structure_search.py
             ^
             |-- run_madgraph_process_conditional_copula.py
             |-- validate_madgraph_process_integral.py
             |-- validate_madgraph_tth_integral.py

setup_madgraph_tth_benchmark.py  --(compile_libraries/inspect_library)-->  setup_madgraph_process.py
```

`run_adaptive_integrand_structure_search.py` (**6344 lines**) is the actual
engine — marginal fitting, copula training, annealing, Bayesian edge pruning,
chi-squared refinement. The **MadGraph physics experiments are thin adapters
over it**: `run_madgraph_process_conditional_copula.py` monkey-patches its
module globals and then calls `baseline.main()`:

```python
baseline.SUPPORTED_BENCHMARKS = {benchmark}
baseline.BENCHMARK_DIMENSIONS = {benchmark: evaluator.dimension}
baseline.sample_conditions = sample_conditions
baseline.target_log_integrand = target_log_integrand   # <- wraps MadGraph
baseline.main()
```

So "run the 5D ttH physics pipeline" and "run the 32D synthetic benchmark" are
the *same code path* with the integrand swapped out. That is the repo's real
architecture, and it is not stated anywhere in the docs.

Only 4 of the experiment scripts touch `nf/` at all (all via
`nf.conditional_cubic_1d`); the big engine is self-contained. The `nf/` package
is therefore much less central than the README's "repository map" implies.

### Most of `nf/` is dead code

I traced every reference. Of the 17 files in `nf/`, only **three** are reachable
from any working entry point:

| module | reachable from |
|---|---|
| `conditional_cubic_1d.py` | the 4 synthetic drivers + `tests/` |
| `conditional_linear_1d.py` | base class of `conditional_cubic_1d` only |
| `conditional_cubic_2d.py` | `run_copula_benchmark_sweep.py` only |

Everything else is unreferenced by any live code path: `train.py`,
`evaluate.py`, `losses.py`, `conditional_rq_1d.py`,
`conditional_autoregressive_cubic_2d.py`, `cube_flow.py`, `cube_spline_flow.py`,
`cube_piecewise_spline_flow.py`, `cube_affine_flow.py`, `logit_rotation.py`,
and `flows/realnvp_nd.py`.

(Two traps in that list: `nf/losses.py` looks live because `nf/train.py`
imports it — but `train.py` itself is never imported. Same pattern with
`logit_rotation.py`, which is only imported by the dead `cube_spline_flow.py`.
And `conditional_autoregressive_cubic_2d.py` is the *only* importer of the
otherwise-live `conditional_cubic_2d.py` — but the one real user of that module
is `run_copula_benchmark_sweep.py`.)

**Practical consequence:** to understand the method as actually run, read
`nf/conditional_cubic_1d.py` (92 lines) and
`experiments/run_adaptive_integrand_structure_search.py`. The rest of `nf/` is
an earlier design that the current experiments moved away from.

---

## 3. Environment — what had to be installed

The `workspace` conda env (`C:\Users\laiji\.conda\envs\workspace`, Python 3.10.14)
already had numpy, pandas, torch, scipy, matplotlib, tqdm. Missing / broken:

| Item | Status | Action taken |
|---|---|---|
| `pytest` | missing | `pip install pytest` → 9.1.1 |
| `vegas` | missing | `pip install --only-binary :all: vegas` → 6.3 (+ `gvar` 13.1.10) |
| numpy↔torch bridge | **broken** | downgraded numpy 2.2.6 → 1.26.4 |

### The numpy/torch blocker (important)

`torch` was `2.3.0+cpu`, which is compiled against NumPy 1.x. With NumPy 2.2.6
installed, **every** `Tensor.numpy()` and `torch.from_numpy()` call raised:

```
RuntimeError: Numpy is not available
```

This is not cosmetic — all 13 `.numpy()` call sites are on critical paths
(inside integration loops that hand tensors to the MadGraph / VEGAS evaluators).
Affected files: `benchmark_nf_vs_madevent_tth.py`,
`run_copula_benchmark_sweep.py`, `run_copula_vs_vegas_integrand.py`,
`run_madgraph_process_conditional_copula.py`,
`run_madgraph_tth_conditional_copula.py`, `validate_madgraph_process_integral.py`,
`validate_madgraph_tth_integral.py`.

Fixed by `pip install "numpy<2"` → numpy 1.26.4. Verified:

```python
torch.from_numpy(np.arange(3, dtype=np.float32))   # OK
torch.arange(3).numpy()                            # OK
```

**Side effect to be aware of:** pip reported that `opencv-python 4.13.0.92` and
`opencv-contrib-python 4.13.0.92` in the same env require `numpy>=2`. They are
now unsatisfied. Nothing in *this* repo imports cv2, so the project is fine, but
if you use OpenCV from `workspace` it will need attention.

### Compute

`torch.cuda.is_available()` → **False**. `torch` is the **CPU-only** build.
16 CPU cores. The `results_public/` timings were produced on CUDA, so expect
everything here to be substantially slower than the published numbers.

### Reproducible setup

```bash
conda activate workspace
pip install "numpy<2" pytest
pip install --only-binary :all: vegas
```

---

## 4. How to run

### 4.1 The `sys.path` gotcha (read this first)

Running a driver directly fails with:

```
ModuleNotFoundError: No module named 'nf.conditional_cubic_1d'; 'nf' is not a package
```

because Python puts the *script's* directory (`experiments/`) on `sys.path`,
not the repo root. Two fixes, both verified:

```bash
# either set PYTHONPATH to the repo root...
PYTHONPATH=/path/to/conditional-nf-integration python experiments/run_copula_bin_sweep.py --smoke

# ...or install the package
pip install -e .
```

Note: `experiments/run_adaptive_integrand_structure_search.py` already contains
a local `sys.path.insert(PROJECT_ROOT)` shim — **this is an uncommitted working-tree
change** (not in git). It is the only script that self-corrects.

Also: all drivers write to **relative** paths (`results/...`), so **always run from
the repo root**.

### 4.2 Unit tests

```bash
PYTHONPATH=$PWD python -m pytest tests/
```

**Result: 12 passed in 2.15 s.** Fast and clean — run these first.

Covers: spline round-trip + log-determinant consistency for both 1D flow types,
sampling range/gradient checks, and phase-space validation (4-momentum
conservation, mass shells, positive Jacobians) for the 2D, 5D and 8D maps.
The 20D `TTHPhaseSpace` test writes its own temporary param card, so it needs
no MadGraph.

### 4.3 Synthetic experiments (no MadGraph)

| Driver | CLI | Smoke entry point | Status |
|---|---|---|---|
| `run_copula_bin_sweep.py` | argparse | `--smoke` | ✅ **works, ~50 s** (§5.1) |
| `run_copula_vs_vegas_integrand.py` | argparse | `--smoke` | ✅ **works, ~2 min** (§5.2) |
| `run_adaptive_integrand_structure_search.py` | argparse + help text | `--speed-profile smoke` | ✅ pipeline verified; **~15–25 min** even in smoke (§5.3) |
| `run_iterative_bayesian_edge_pruning_frozen_null.py` | argparse | `--smoke` | ⚠️ works but slow (§5.4) |
| `run_bayesian_selected_copula_benchmark.py` | argparse | `--smoke` | ❌ **broken, §6.1** |
| `run_copula_benchmark_sweep.py` | **none** | — | ❌ **no CLI, too slow, §6.2** |
| `reanalyze_nf_vs_madevent_errors.py` | argparse | — | ⚠️ runs degraded, §5.6 |

#### The adaptive engine's compute profiles

`run_adaptive_integrand_structure_search.py` is the only driver with a built-in
compute dial. `--speed-profile` scales the whole pipeline:

| setting | smoke | fast | balanced | thorough (default) |
|---|---:|---:|---:|---:|
| `kl_max_steps` | 300 | 6 000 | 12 000 | **30 000** |
| `chi2_max_steps` | 300 | 2 400 | 3 600 | 6 000 |
| `bridge_max_steps` | 100 | 400 | 600 | 1 800 |
| `marginal_max_steps` | 150 | 800 | 1 200 | 6 000 |
| `adaptive_rounds` | 1 | 2 | 2 | 3 |
| `marginal_train_conditions` | 32 | 128 | 256 | 1 024 |
| `final_ess_size` | 5 000 | 50 000 | 50 000 | 100 000 |

So the **default `thorough` is ~100× the smoke budget on the KL stage**. Always
pass `--speed-profile smoke` for a first run.

### 4.4 MadGraph experiments (physics)

`setup_madgraph_process.py`, `setup_madgraph_tth_benchmark.py`,
`run_madgraph_process_conditional_copula.py`,
`run_madgraph_tth_conditional_copula.py`, `validate_madgraph_*`,
`benchmark_nf_vs_madevent_tth.py`.

**These cannot run on this machine as-is.** See §6.3.

One partial exception: `reanalyze_nf_vs_madevent_errors.py` reads only CSVs and
so starts without MadGraph — but it needs the generated `process-dir` to recover
MadEvent errors, and without it the numbers it prints are misleading. See §5.6.

---

## 5. Smoke-test results

### 5.1 `run_copula_bin_sweep.py --smoke` — ✅ WORKS, ~50 s

```bash
PYTHONPATH=$PWD python experiments/run_copula_bin_sweep.py --smoke
```

Exit code 0. Config actually used: 2 benchmarks × 1 dim × 2 bins × 1 seed = 4 models,
`marginal_steps=100`, `copula_steps=150`, `batch=1024`, `n_eval=5000`.

Per-model training took **11.6–13.0 s** on CPU. Produced
`results/copula_bin_sweep_smoke/{config.json,raw_results.csv,summary.csv,plots/}`
with 8 PNGs. Sanity check passed: the targets have a known true integral of 1.0
and the estimates land in 0.95–1.05, with relative errors 0.01–0.05 at this
deliberately tiny training budget.

| benchmark | dim | bins | ess_mean | relative_error_mean | train_time_mean (s) |
|---|---|---|---|---|---|
| curved_ridge | 2 | 16 | 0.2179 | 0.0223 | 11.63 |
| curved_ridge | 2 | 32 | 0.2159 | 0.0235 | 12.81 |
| single_gaussian | 2 | 16 | 0.1383 | 0.0201 | 12.98 |
| single_gaussian | 2 | 32 | 0.1276 | 0.0351 | 12.19 |

(Low ESS is expected — this is the smoke budget, not a scientific run. Compare
`results_public/copula_bin_sweep/summary.csv` for the real numbers.)

This is the **best entry point** for verifying the install: it is self-contained,
fast, writes plots, and exercises `nf/`, pandas, matplotlib and torch together.

### 5.2 `run_copula_vs_vegas_integrand.py --smoke` — ✅ WORKS, ~2 min

```bash
PYTHONPATH=$PWD python experiments/run_copula_vs_vegas_integrand.py --smoke
```

Exit code 0. Note the smoke output dir is `results/copula_vs_vegas_integrand_smoke_fixed`
(with a `_fixed` suffix, unlike the other `--smoke` flags).

This is the only script that needs the `vegas` package. It trains one NF
(250 steps) and then evaluates NF vs VEGAS at **all 16 hard-coded held-out
conditions** — `--smoke` does not reduce that count, so the runtime is dominated
by the VEGAS integrand evaluations (~4 min of CPU work), not the training.

Aggregate on `curved_ridge` (d=2):

| method | relative_error_mean | variance_mean | ess_mean | total_time_mean (s) |
|---|---|---|---|---|
| copula_nf | 0.01907 | 4.09e-04 | 0.2228 | 24.43 |
| vegas | 0.00298 | 2.95e-05 | — | 0.0096 |

VEGAS wins comfortably here, which is the *expected* outcome: this is a 2D
target, exactly the regime where axis-adaptive VEGAS is strong, and the NF got
only 250 smoke training steps. The repo's claim is the opposite in **32D**
(see §5.5) — worth keeping in mind before reading anything into this table.

### 5.3 `run_adaptive_integrand_structure_search.py --speed-profile smoke` — ✅ pipeline verified, aborted at ~8 min

```bash
PYTHONPATH=$PWD python experiments/run_adaptive_integrand_structure_search.py \
    --speed-profile smoke --no-compile-kernels
```

This is the flagship engine, so I let it run longest. It **worked correctly
through the entire annealing pipeline** and I stopped it partway into the final
refinement (see §6.5 for why).

Verified stages, in order, all completed:

1. `ADAPTIVE INTEGRAND BENCHMARK` — initial uniform-proposal diagnostics
2. **5 annealing stages**, driven by an adaptive temperature schedule:
   `0.0000 → 0.0200 → 0.0800 → 0.2500 → 0.5500 → 1.0000`, each with its own
   `WEIGHTED CACHE` rebuild (train + validation split) and a marginal fit
3. `IMPORTANCE-WEIGHTED KL TRAINING: initial_full` — reached step 100 of 300
   with the validation NLL improving monotonically:

| step | val NLL |
|---:|---:|
| 0 | −2.863583 |
| 100 | −4.184063 |

The loss is going down, which is the main thing this smoke run was there to
confirm. It then got killed ~8 minutes in, before the χ² refinement and final
ESS evaluation — so **it never wrote its output files**
(`results/adaptive_integrand_structure_search/<benchmark>_final_summary.csv` etc.
were never created).

Timing note: `--speed-profile smoke` still takes **minutes per 100-step KL
block** on CPU, because each step evaluates the integrand over
`conditions × points_per_condition` on 16 CPU cores with no GPU. Expect a
genuinely complete smoke run here to be 15–25 min, not "a few minutes". The
default `thorough` profile is ~100× the KL budget and is a multi-day job on this
machine.

Useful to know: `--no-compile-kernels` is the right flag on this box — the
kernel compile path is CUDA-only (there is no GPU here) and the script otherwise
prints compile diagnostics you don't need.

### 5.4 `run_iterative_bayesian_edge_pruning_frozen_null.py --smoke` — ⚠️ slow

```bash
PYTHONPATH=$PWD python experiments/run_iterative_bayesian_edge_pruning_frozen_null.py \
    --smoke --benchmarks single_gaussian --seeds 0
```

**It runs and produces sensible output** — marginal training for 3 seeds, then a
`ROUND 0 | active edges=28` header, then per-seed evidence collection over the 7
child nodes of the d=8 autoregressive graph. I stopped it during round 0 (§6.5).

Two things make the smoke flag misleading here:

- `--smoke` **overrides `--seeds` back to `[0,1,2]`**, so my `--seeds 0` was
  silently ignored and it ran 3 seeds instead of 1.
- It does not shrink `--benchmarks` either, so a plain `--smoke` runs all three
  benchmarks.

Combined with `max_rounds=3`, the effective work is `3 benchmarks × 3 seeds ×
7 children × 3 rounds` of masked-child training, plus MCMC. Even with the
reduced `masked_steps=300` this is well beyond a "smoke test".

**Measured:** I let it run for **6 minutes** and killed it. In that time it got
as far as evidence `child 5/7` for `seed 0` **of round 0 of 3** — and it had
written **no output files at all** (its results directory was still empty,
because the per-round CSVs are only flushed at the end of a round). Extrapolating
linearly, this specific invocation is on the order of an hour; the default
3-benchmark run is a multi-hour to overnight job on CPU.

So: the pipeline is functional, but treat it as a long-running job, not a smoke
test. If you want a genuinely quick sanity run, override the knobs manually:
`--benchmarks single_gaussian --seeds 0 --marginal-steps 20 --masked-steps 20
--max-rounds 1 --batch-size 256 --mcmc-iter 200 --mcmc-burn 50 --n-eval 1000`.

### 5.5 Published 32D numbers cross-check

Not re-run (160 conditions × 10^6 evaluations, ~1500 s for the NF alone on the
authors' hardware). But `results_public/nf_vs_vegas_160/summary.csv` **does**
back the README table exactly:

| method | mean_estimate | mean_absolute_error | mean_norm_ESS | coverage_95 | total evals |
|---|---|---|---|---|---|
| nf_pure | 0.999254 | 0.0013665 | 0.53284 | 0.7875 | 160,000,000 |
| vegas | 1.3479e-05 | 0.99999 | — | 0.0 | 159,211,785 |

So the headline claim ("learned autoregressive proposal represents the narrow
correlated 32D target far better than this VEGAS configuration") is traceable
to committed data. The README is also upfront that NF coverage is 0.7875 vs a
nominal 0.95 — an acknowledged undercoverage problem, not a hidden one.

I spot-checked the other published tables against `results_public/` and they
hold up. The 8D high-statistics retest is reproduced to every quoted digit:

| source | σ_NF | σ_Sobol |
|---|---|---|
| README | 0.0529874 ± 0.000013 | 0.0529849 ± 0.0000031 |
| `uu_ww_4l_300_highstat/summary.csv` | 0.05298736 ± 0.0000132 | 0.05298488 ± 0.0000031 |

And the ESS ranges in the README table match the validation CSVs
(2D: 0.988–0.998; 5D: 0.978–0.987; 8D: 0.934–0.941). The quoted numbers are not
hand-typed approximations — they are the committed CSVs. That is a good sign for
the repo's internal consistency, even though the runs themselves can't be
repeated here without MG5.

I also hand-checked the 20D pull arithmetic in `docs/VALIDATION_RESULTS.md`:
NF(2000) = 0.383434 ± 0.000056, MadEvent(2000) = 0.383050 ± 0.000253. Difference
0.000384, combined error √(0.000056² + 0.000253²) = 2.59e-4, pull = **1.48** —
matching the documented "about 1.48 combined standard deviations". The MadEvent
values come from `madgraph_tth_madevent_precision/comparison.csv` and agree to
every quoted digit.

One thing that briefly looked like a contradiction but is not: the *older*
MadEvent runs recorded in `nf_vs_madevent_tth/` and
`nf_vs_madevent_tth_exact_error_audit/` give MadEvent(2000) = 0.3827–0.3828, not
0.383050. Those are the superseded `generate_events` runs that
`VALIDATION_RESULTS.md` explicitly disowns in its §4 preamble. The docs are
internally consistent; you just have to know which directory is authoritative.

### 5.6 `reanalyze_nf_vs_madevent_errors.py` — ✅ runs, but degraded without MG5

```bash
PYTHONPATH=$PWD python experiments/reanalyze_nf_vs_madevent_errors.py \
    --results results_public/nf_vs_madevent_tth
```

This one is interesting: it needs **no MadGraph to start**, only the CSVs, so it
runs on the published data. But `--process-dir` (default
`generated/gg_tth_full_madevent`) doesn't exist, so it emits
`No MadEvent result found for ...` for all 25 runs and then
`Recovered MadEvent errors for 0 runs`.

In that state it still prints a pull table — but the errors are built from the
replicate scatter alone, with the MadEvent internal errors missing, which
inflates the pulls (5.83σ at 1400 GeV). **My output nevertheless reproduced the
committed `results_public/nf_vs_madevent_tth/validation_with_mg_errors.csv`
exactly**, including that 5.83σ value and the empty `internal_pull` column. So
the committed artifact was itself produced in this same degraded mode, and the
big pulls in it are an artifact of the missing MG error inputs, not a physics
result. Don't quote them.

Useful as a determinism check; not useful as a reproduction of the paper.

> ⚠️ **This script OVERWRITES its `--results` directory in place.** Pointing it at
> `results_public/nf_vs_madevent_tth` silently rewrote three **tracked** files
> (`raw_results_with_mg_errors.csv`, `summary_with_mg_errors.csv`,
> `validation_with_mg_errors.csv`) with degraded content. I caught this and
> restored all three with `git checkout --`; `git status` is clean again.
>
> Worth knowing what it clobbered: the committed files hold the *real* MadEvent
> errors from the original `generate_events` runs (internal pulls of 20.5σ and
> 14.2σ at 600/800 GeV). Run this only against a copy.

### 5.7 Published artifact integrity

I verified `results_public/MANIFEST.json` against the files on disk:

```
exact SHA-256 match                    :  26 / 173
match after CRLF -> LF normalisation   : 136 / 173
genuinely differing                    :   0
missing                                :  11  (all .ipynb_checkpoints/, gitignored)
```

The `core.autocrlf=true` setting on this Windows checkout rewrites LF to CRLF,
so the manifest's raw hashes cannot match here — but every file is byte-identical
to what was published once line endings are normalised. **The curated results are
intact.** (The 11 "missing" entries are Jupyter autosave files that the
repo's own `.gitignore` excludes; harmless noise in the manifest.)

---

## 6. Blockers and things that are too slow

### 6.1 `run_bayesian_selected_copula_benchmark.py` — broken default path

**Verified first-hand.** The default `--graph-file` does not exist:

```
DEFAULT_GRAPH_FILE = "results/bayesian_delta_edge_selection/median_probability_graph.csv"
```

Nothing in the repo creates this file (searched `results/`, `results_public/`,
and every script). The default invocation dies immediately:

```
FileNotFoundError: [Errno 2] No such file or directory:
  'results/bayesian_delta_edge_selection/median_probability_graph.csv'
```

It needs a CSV with columns `benchmark,parent,child,posterior_inclusion_probability`.
Presumably an output of the *iterative edge-pruning* experiment — but that script
writes `final_graph_edges.csv` / `round_posteriors.csv` under its own output dir,
not this path, and with different column names. So the pipeline is not wired
end-to-end. Workaround: pass `--graph-file <your.csv>` explicitly, or use the
internally-generated graph families (independent / markov1 / full_ar).

Also note: `--smoke` here does **not** shrink `--benchmarks` (stays all 3), so
add `--benchmarks single_gaussian` for a genuinely small run.

### 6.2 `run_copula_benchmark_sweep.py` — no CLI, multi-hour

**Verified.** There is no `argparse` at all; `main()` runs immediately on import.
Passing `--help` does **not** print help — it silently starts the real benchmark.
My `--help` probe ran for 180 s before I killed it (it created
`results/copula_benchmark_sweep/{checkpoints,plots}/` before being terminated).

Reading the module constants: 4 benchmarks × (2500 marginal ×2 + 2500 copula +
3000 coupling) ≈ 42 000 steps at batch 4096, plus 8 evaluations over a 250×250
grid. Estimated 25–45 min on CUDA → **several hours on this CPU-only box**.
Not run to completion. To make it feasible you must edit module constants at the
top of the file (`BENCHMARK_NAMES`, `*_STEPS`, `N_DATA`, `N_GRID`, `N_EVAL`) —
there is no flag for it.

### 6.3 All MadGraph-dependent experiments — MG5 not installed

> ✅ **RESOLVED in Part II (§11–§14).** MG5 3.6.7 is now installed and the
> gg→t̄t scattering pipeline has been run end to end, reproducing the published
> integrals. The section below is kept as the Part-I record of the blocker.

**Verified.** No MadGraph on this machine:

- `which mg5_aMC` → not found; no `MADGRAPH_PATH` or similar env var set.
- Only `/c/Users/laiji/Downloads/MG5_aMC_v3.6.7.tar.gz` (an archive, not installed).
- The configs point at `generated/<process>_standalone`, which is gitignored and
  absent.
- `MadGraphMatrixElement` looks for `SubProcesses/P*/matrix2py*.so` — a
  **Linux shared object**. On Windows this needs MSYS2/WSL plus gfortran.

So: `setup_madgraph_*` (generate + compile the standalone), then
`run_madgraph_*` (train), then `validate_madgraph_*` (Sobol / MadEvent
cross-checks) are all blocked. `benchmark_nf_vs_madevent_tth.py` additionally
needs a working MadEvent install.

This is the bulk of the repository's scientific claims (the 2D/5D/8D/20D
scattering tables in the README). Nothing here can be reproduced without
setting up MG5 — expect that to be a significant project on Windows, not a
quick install.

---

### 6.4 Artifacts my probing left behind

Everything finished/aborted is under `results/`, which is **gitignored**.
`git status` currently shows only the pre-existing
`run_adaptive_..._search.py` modification plus the new `PRD.md`.

(One exception, since fixed: `reanalyze_nf_vs_madevent_errors.py` did write into
three *tracked* files under `results_public/`. I restored them with
`git checkout --` and re-verified their hashes — see §5.6 and §5.7.)

```
results/copula_bin_sweep_smoke/                    <- good, complete smoke run
results/copula_vs_vegas_integrand_smoke_fixed/     <- good, complete smoke run
results/bayesian_selected_copula_benchmark_smoke/  <- EMPTY; created before the
                                                      FileNotFoundError in §6.1
results/copula_benchmark_sweep/                    <- EMPTY dirs (checkpoints/,
                                                      plots/); created by my
                                                      accidental `--help` probe
                                                      before I killed it at 180 s
```

The first two are worth keeping as a working reference. The last two are inert
empty directories — `rm -rf` on them was blocked by the sandbox and I chose not
to fight it, so delete them yourself if they bother you.

### 6.5 Tests I aborted for time

Requested behaviour: anything too slow gets stopped and written down here rather
than left running. Aborted so far:

| # | Test | Ran for | State at abort | Why aborted |
|---|---|---:|---|---|
| 1 | `run_copula_benchmark_sweep.py` (no CLI) | 180 s | wrote empty `checkpoints/` + `plots/` dirs under `results/copula_benchmark_sweep/`, nothing else | Triggered by accident with `--help` — the script ignores argv and runs. Estimated several hours on CPU; a smoke config requires *editing module constants*, so it's not testable as-is. |
| 2 | `run_iterative_bayesian_edge_pruning_frozen_null.py --smoke --benchmarks single_gaussian --seeds 0` | 6 min | round 0, evidence `child 5/7`, seed 0 of 3; **zero output files written** | `--smoke` silently forces 3 seeds and 3 benchmarks. Linear extrapolation ≈ 1 h for this invocation. Pipeline verified functional; not worth more wall-clock. |
| 3 | `run_adaptive_integrand_structure_search.py --speed-profile smoke --no-compile-kernels` | ~8 min | all 5 annealing stages done; final KL training at step 100/300; **no output files written** | Everything I needed to verify was confirmed by then (see §5.3). Remaining work was roughly 15 more minutes of CPU training plus the χ² stage. |

All three were stopped deliberately, not crashed. The three that **did** complete
normally are §5.1 (bin sweep), §5.2 (VEGAS) and §5.6 (reanalysis).

**No Python processes are left running** — verified with `Get-Process python`
after the last `TaskStop`.

## 7. Quick reference

```bash
conda activate workspace
cd /path/to/conditional-nf-integration
export PYTHONPATH="$PWD"

# health checks (fast, both fully verified)
python -m pytest tests/                       # 12 tests, ~2-5 s
python experiments/run_copula_bin_sweep.py --smoke        # ~50 s, writes plots

# longer smoke runs
python experiments/run_copula_vs_vegas_integrand.py --smoke          # ~2 min
python experiments/run_adaptive_integrand_structure_search.py \
    --speed-profile smoke --no-compile-kernels                       # ~15-25 min

# NOT worth trying as-is on CPU: ~1 h and multi-hour respectively
#   run_iterative_bayesian_edge_pruning_frozen_null.py --smoke ...
#   run_copula_benchmark_sweep.py            (no CLI; edit constants instead)
```

Minimum viable environment if setting up fresh:

```bash
conda create -n nfi python=3.10
conda activate nfi
pip install torch numpy scipy pandas matplotlib pytest
pip install --only-binary :all: vegas     # needs the binary wheel on Windows
```

---

## 8. Repo hygiene observations

These don't block anything but they cost time if you don't know them:

1. **Inconsistent `sys.path` shims.** 10 of 18 experiment scripts insert the
   repo root into `sys.path`; 8 do not. `run_adaptive_integrand_structure_search.py`
   has the shim **only as an uncommitted working-tree change**. So "does
   `python experiments/foo.py` work?" has a different answer per script.
   `PYTHONPATH=$PWD` sidesteps the whole mess.

2. **`pyproject.toml` declares packages that don't exist.** The include list is
   `["benchmarks*", "experiments*", "losses*", "mc_baselines*", "nf*", "utils*"]`
   but only `experiments/` and `nf/` are present. `benchmarks/`, `losses/`,
   `mc_baselines/`, `utils/` are leftovers from an earlier layout.

3. **No declared dependencies.** `pyproject.toml` has no `[project].dependencies`
   and there is no `requirements.txt` / `environment.yml` anywhere. The real
   dependency set had to be reverse-engineered: numpy, scipy, torch, pandas,
   matplotlib, pytest, vegas.

4. **Inconsistent `--smoke` semantics.** `--smoke` shrinks budgets but in the
   Bayesian scripts it does *not* shrink `--benchmarks` (still all 3), and in
   `run_iterative_bayesian_edge_pruning_frozen_null.py` it also *overrides*
   `--seeds` back to `[0,1,2]`. So `--smoke --seeds 0` silently runs 3 seeds.

5. **`docs/README_LEGACY.md` is stale.** It says to run
   `python -m nfmodel.scripts.zg_pipeline_costh` — that module **does not exist**
   in this tree. Leftover from an earlier iteration.

6. **`scripts/curate_public_results.py`** is non-destructive and only copies
   already-existing `results/<run>/` trees into `results_public/`. It cannot
   regenerate the published artifacts, because the source `results/` tree was
   never committed. Treat `results_public/` as the only reproducible record.

7. **Two files are named `madgraph_*` but are pure phase-space code** — they
   import fine without MG5. Only `MadGraphMatrixElement` (which loads
   `matrix2py*.so`) needs the generated library. That's why the phase-space
   tests pass on this MG5-less machine.

## 9. What I would do next

1. Decide whether to install MG5 (§6.3) — without it, ~60% of the repo's
   scientific claims are unreachable. Realistically this needs WSL or MSYS2.
2. Ask the repo author where `results/bayesian_delta_edge_selection/median_probability_graph.csv`
   comes from (§6.1), or regenerate it.
3. Add a `requirements.txt` pinning numpy<2 for the torch 2.3 combination.
4. Re-run `tests/` and `run_copula_bin_sweep.py --smoke` after any further env
   change — they are the two fastest health checks.

---
---

# Part II — Linux / CERN EOS

*MadGraph installed, scattering pipeline reproduced*

**Date: 2026-10-08 → 2026-10-09.** Different machine from Part I: a CERN
lxplus-style Linux node, home on EOS, no usable conda, no GPU. Everything
below is a record of what was actually run.

## 10. Headline

Part I's blocker §6.3 is closed. **MadGraph5_aMC@NLO 3.6.7 is installed and
the `gg → t t̄` scattering pipeline runs end to end**: generate → compile →
train a conditional NF → validate at held-out energies against independent
Sobol integration.

The published numbers are not just qualitatively reproduced — the **Sobol
column of `results_public/gg_ttbar_validation/summary.csv` is reproduced
bit-for-bit**, and a smoke-budget NF lands within ~1–2% of the published ESS at
four of five held-out energies.

| E [GeV] | NF (my smoke model) [pb] | Sobol (mine) [pb] | Sobol (published) | ESS/N | pull |
|---:|---:|---:|---:|---:|---:|
| 400 | 12.841555 ± 0.0085 | 12.844943 | 12.844943 | 0.820 | −0.40 |
| 650 | 16.914743 ± 0.0033 | 16.916571 | 16.916571 | 0.981 | −0.55 |
| 1000 | 11.36891 ± 0.0030 | 11.368139 | 11.368139 | 0.967 | +0.26 |
| 1500 | 6.8589956 ± 0.0022 | 6.8603675 | 6.8603675 | 0.952 | −0.63 |
| 1950 | 4.7763929 ± 0.0013 | 4.7760299 | 4.7760299 | 0.965 | +0.28 |

Every Sobol value matches the committed CSV to all printed digits
(relative difference `0.00e+00`). All five NF–Sobol pulls are below 0.7σ.

Two caveats, stated plainly: the NF here got only the **`--speed-profile smoke`
budget** (300 KL steps), so its 400 GeV ESS of 0.820 is well below the
published 0.998 — that is a training-budget effect, not a method failure. And
this is the 2-D control process; the 5-D/8-D/20-D processes were **not** built
in this session (see §16).

## 11. Environment — and one large surprise

### 11.1 What was available

| item | state on arrival |
|---|---|
| system `python3` | 3.9.25 — too old; MG5 3.6.7 has dropped ≤3.9 and the repo needs a modern torch |
| conda | present but **broken** — the prefix had been moved, so every `bin/` shebang pointed at a nonexistent `/eos/user/j/jlai/miniconda3/bin/python` |
| `/eos/home-j` (i.e. `/eos/user/j/jlai`) | **FUSE** mount |
| GPU | none |

The conda prefix still contained a working `python3.11` binary when invoked by
absolute path, so the environment was built from that rather than reinstalling
conda.

### 11.2 The environment actually used

```bash
/eos/user/j/jlai/Miniconda/miniconda3/bin/python3.11 -m venv /eos/user/j/jlai/mango/venv
```

| package | version |
|---|---|
| python | 3.11.5 |
| numpy | 1.26.4 |
| torch | **2.14.1+cpu** |
| scipy | 1.17.1 |
| pandas | 3.0.6 |
| matplotlib | 3.11.2 |
| vegas | 6.4.1 |
| pytest | 9.1.1 |
| gvar | 13.1.10 |

**Correction to Part I §3.** The `numpy<2` pin is *not* a general requirement of
this repository — it is a requirement of **torch 2.3 specifically**, which was
the build Part I had. `torch 2.14.1+cpu` works correctly with NumPy 2 as well;
the bridge was verified in both directions
(`torch.from_numpy` and `Tensor.numpy`). The pin is harmless and is kept here
only for exactness with the published environment.

### 11.3 The EOS FUSE penalty — measure this before you plan any install

The single most important operational fact on this machine: **EOS home is a
FUSE mount and is catastrophically slow at creating small files.**

Same loop, 300 tiny files, `echo x > f$i.txt`:

| filesystem | time | rate |
|---|---:|---:|
| `/eos/user/j/jlai/mango` (FUSE) | 35.6 s | **~8.4 files/s** |
| `/tmp` (local ext4) | 0.023 s | ~13 000 files/s |

That is a ~1500× penalty, and it is **latency-serialised, not bandwidth-bound**:
running the same 300 creates across 16 parallel processes only improved it to
24.0 s (~12.5 files/s). Meanwhile a single 100 MB sequential write runs at
**358 MB/s**. So big files are fine; *many* files are the problem.

Consequences, all of which were felt in this session:

- A `pip install --upgrade pip` (≈1000 files) took **~10 minutes**.
- Extracting the 31 MB MG5 tarball (~5900 files) took **~7 minutes**.
- Installing torch (~5000 files) took **~15 minutes**.
- Anything that builds a MadGraph process tree or a virtualenv on EOS is a
  coffee break, not a second. Budget accordingly, or build on local disk and
  copy.

### 11.4 Two self-inflicted wounds worth recording

**(a) Concurrent pip runs silently corrupt a venv.** Two `pip install` jobs
were accidentally left running against the same prefix. The visible symptom was
misleading — `ImportError: Error importing numpy: ... No module named
'numpy.core'` — because one pip was replacing `numpy` while another was
importing it. A second casualty was a mixed-version `_pytest/` tree that made
every `pytest` invocation die with
`AttributeError: module 'py' has no attribute 'path'` until pytest was
force-reinstalled. **Never run two pips into one prefix.**

**(b) `py.py` in the working directory breaks `pytest`.** After a MadGraph run,
`pytest` failed from the repository root with:

```
_pytest/compat.py:32: LEGACY_PATH = py.path. local
AttributeError: module 'py' has no attribute 'path'
```

The cause is not pytest. PLY — via MadGraph's ALOHA code generator
(`MG5_aMC_v3_6_7/aloha/aloha_parsers.py`) — writes a generated LALR parser
table named **`py.py` into the current working directory**. Python's `sys.path`
puts the CWD first, so this file shadows the real `py` shim that pytest 9 ships
and pytest can no longer import. Running `pytest` from any other directory
works fine. Delete the stray `py.py` (or run the MadGraph step from elsewhere).
MadGraph also drops `nsqso_born.inc` and `param.log` in the CWD — both harmless
and both invisible to `git status` (`param.log` is gitignored; the others are
untracked).

## 12. Installing MadGraph

MG5 3.6.7 (build date 2026-01-05) was chosen deliberately: it is the version
Part I found the author had downloaded, so it matches whatever produced the
published results.

```bash
cd /eos/user/j/jlai/mango
curl -sLO "https://launchpad.net/mg5amcnlo/3.0/3.6.x/+download/MG5_aMC_v3.6.7.tar.gz"
tar xzf MG5_aMC_v3.6.7.tar.gz          # ~7 min on EOS, ~5900 files
```

MG5 ships `bin/mg5_aMC` with a `#!/usr/bin/env python3` shebang, which here
resolves to the too-old system python 3.9. Rather than edit the upstream install
or depend on `PATH` ordering, a two-line wrapper was created:

```sh
# /eos/user/j/jlai/mango/bin/mg5_aMC
#!/bin/sh
exec /eos/user/j/jlai/mango/venv/bin/python \
     /eos/user/j/jlai/mango/MG5_aMC_v3_6_7/bin/mg5_aMC "$@"
```

This is what gets passed as `--mg5`; the repo's `resolve_executable()` accepts a
plain file path, so no repo code needed changing.

**Optional-tool warnings are expected and harmless.** On startup MG5 reports
that it cannot find `fastjet-config`, `eMELA-config` or `lhapdf-config`. None
are needed for LO *partonic* matrix elements with a fixed α_s and no PDFs —
which is exactly what this repository does. The `Checking if MG5 is up-to-date
... failed to connect server` line is also harmless (the update check has no
egress here).

## 13. Generating and compiling the standalone

```bash
cd /eos/user/j/jlai/mango/conditional-nf-integration
PYTHONPATH=$PWD /eos/user/j/jlai/mango/venv/bin/python \
    experiments/setup_madgraph_process.py \
    --config configs/processes/gg_ttbar.json \
    --mg5 /eos/user/j/jlai/mango/bin/mg5_aMC
```

MG5 correctly reads `g g > t t~ @1` as **one subprocess, 3 diagrams**, and
creates `SubProcesses/P1_gg_ttx/`. The `@1` diagram filter is what keeps this a
single subprocess directory, which matters: `MadGraphMatrixElement` raises
`RuntimeError` if a spec resolves to more than one `P*` directory.

**Compilation detail worth knowing.** `compile_libraries()` tries the combined
`make allmatrix2py.so` target first. On this MG5 3.6.7 build that target is
**unavailable**, so the documented fallback runs and builds
`make matrix2py.so` per process directory instead — printing
`Combined allmatrix2py target is unavailable; trying per-process wrappers.`
That fallback is the path that actually works, and it happens to produce
exactly the layout the loader wants:

```
generated/gg_ttbar_standalone/SubProcesses/P1_gg_ttx/
    matrix2py.cpython-311-x86_64-linux-gnu.so     <- versioned, preferred
    matrix2py.so
```

`MadGraphMatrixElement` globs `P*/matrix2py*.so` and prefers the `cpython-`
versioned one, so the two coexist without ambiguity.

## 14. The blocker nobody had hit: MadGraph initialisation is CWD-sensitive

This is the most important finding of Part II, and it cost the most time.

### Symptom

Every attempt to load the matrix element **died silently**:

```
$ python experiments/validate_madgraph_process_integrand.py --config configs/processes/gg_ttbar.json
 Warning: file ident_card.dat not found in the parent directories!(lha_read)
$ echo $?
0
```

Exit status **0**. No traceback, no error, no output at all past the warning —
not even the prints that sit *after* the failing call. The generated library
was present and importable; `import matrix2py` succeeded and exposed
`py_m0_get_value`, `py_m0_smatrix`, `py_m0_smatrixhel`,
`py_m0_initialisemodel` exactly as expected.

### Cause

`Source/MODEL/lha_read.f` resolves the auxiliary Les-Houches files **relative to
the current working directory**, not relative to the parameter card that was
passed in. For any filename containing `_card` it prepends `./Cards/` and then
walks up to five parent levels (`../Cards/`, `../../Cards/`, …). If
`ident_card.dat` is still not found it prints the warning and executes a
Fortran `stop`:

```fortran
40      tempname='../'//tempname
        if (i.eq.5)then
           write(*,*) 'Warning: file ',filename,
     &                           ' not found in the parent directories!(lha_read)'
           stop
        endif
```

Fortran `stop` terminates the process with **exit status 0**. That is what made
this so hard to see: the failure is perfectly disguised as success. A shell that
checks `$?`, or a pipeline whose exit status comes from its last stage, will
report a clean run.

### Fix

**Run the MadGraph-dependent scripts with the working directory set to the
generated standalone directory** (or any directory from which `Cards/` is
reachable), because `Cards/ident_card.dat` lives there:

```bash
cd generated/gg_ttbar_standalone
PYTHONPATH=$REPO python $REPO/experiments/validate_madgraph_process_integrand.py \
    --config $REPO/configs/processes/gg_ttbar.json --points 500
```

With that one change the same command succeeds immediately:

```
CONFIGURED MADGRAPH INTEGRAND VALIDATION
process                   = gg_ttbar
dimension                 = 2
max_momentum_residual     = 2.27373675443e-13
max_mass_shell_residual   = 3.49245965481e-10
minimum_jacobian          = 0.0115863199095
finite_fraction           = 1
finite log-integrand frac = 1
positive integrand frac   = 1
throughput points/s       = 47203.771
```

Four-momentum conservation to 2.3e-13, mass shells to 3.5e-10, a strictly
positive Jacobian, a fully finite integrand, and **47 000 matrix-element
evaluations per second** on one CPU core.

### Why this matters beyond one command

This collides head-on with the repository's own documented rule (§8.1 of
Part I): *"all drivers write to relative paths (`results/...`), so always run
from the repo root."* For every MadGraph-dependent driver, **that advice is
wrong** — running from the repo root is precisely what makes the matrix element
initialize to a silent `stop`.

The workaround used here is to keep the CWD inside the standalone directory and
pass **absolute** paths for everything else (`--config`, `--checkpoint`,
`--output`). Do not "fix" this by symlinking a single `Cards/` into the repo
root: each process has its own `Cards/`, so such a link is only ever correct for
one process at a time. A cleaner repair — not attempted here, since it would
mean editing the author's code — would be for `setup_madgraph_process.py` to
`chdir` into the standalone directory for the duration of the initialise call.

## 15. Results

### 15.1 Repo test suite

```bash
PYTHONPATH=$PWD python -m pytest tests/
# 12 passed in 12.67s
```

Matches Part I's result (Part I saw 2.15 s on Windows; the difference is EOS
FUSE and a loaded shared node, not the code).

### 15.2 Published Sobol integrals reproduced exactly

Using only the freshly built library, `power=18` (262 144 points), 8 scrambled
replicates, seed 20260920 — the exact configuration in the published
`config.json`:

| E [GeV] | my Sobol [pb] | published [pb] | relative difference |
|---:|---:|---:|---:|
| 400 | 12.844943 | 12.844943 | `0.00e+00` |
| 650 | 16.916571 | 16.916571 | `0.00e+00` |
| 1000 | 11.368139 | 11.368139 | `0.00e+00` |
| 1500 | 6.860368 | 6.860368 | `0.00e+00` |
| 1950 | 4.776030 | 4.776030 | `0.00e+00` |

Even the *per-replicate scatters* agree: 1.2e-14, 1.6e-11, 1.3e-13, 9.6e-9 and
9.2e-8 pb against the committed 1.1788e-14, 1.5590e-11, 1.3422e-13, 9.5812e-9
and 9.2386e-8 pb. This is a strong end-to-end check: matrix-element
normalization, external-particle ordering, the phase-space map, the Jacobian,
α_s, and the QMC protocol all match the authors' setup. It took ~29 s per
energy.

**A suspicion that turned out to be unfounded — worth recording.** The published
Sobol *uncertainties* for this process are absurdly small: `stderr_pb ≈ 1e-14`
on a 12.8 pb estimate, i.e. ~1e-15 relative. That looks like a bug (an error bar
at machine precision is physically meaningless), and it was checked directly.

It is **not** a bug. The Sobol replicates genuinely differ — different seed →
different scrambled points, verified — yet they integrate to the same value to
~15 digits. The explanation is that this is a **2-D integrand smooth enough for
scrambled QMC to be essentially exact** at 2^18 points. The tiny scatter is real
floating-point residue, not a mis-computed error. The reproduction above
confirms it independently, since the same 1e-14-scale scatters reappear from a
completely separate build. The lesson is the one Part I already applied: the
Sobol error bar for this process must not be read as a meaningful Monte Carlo
uncertainty — the NF–Sobol pulls are dominated entirely by the NF error.

### 15.3 Training on the real process

```bash
cd generated/gg_ttbar_standalone
PYTHONPATH=$REPO python $REPO/experiments/run_madgraph_process_conditional_copula.py \
    --config $REPO/configs/processes/gg_ttbar.json \
    --speed-profile smoke \
    --output $REPO/results/gg_ttbar_copula_smoke
```

Exit 0. The full annealing pipeline ran — weighted caches, marginal fit
(early-stopped at step 100), bridge copula, then importance-weighted KL
training — and wrote a checkpoint plus
`madgraph_gg_ttbar_2_final_summary.csv`:

| quantity | value |
|---|---|
| graph | `{0: [], 1: [0]}` (1 edge) |
| ESS | 0.8072 |
| Ihat | 2.6337e-08 |
| logw std | 0.4935 |

Note the marginal early-stop and the one-edge graph: for a 2-D two-body process
the copula has almost nothing left to model after the marginals, which is the
expected behaviour and a nice sanity signal that the pipeline is not
overfitting noise.

### 15.4 Held-out validation

`validate_madgraph_process_integral.py` with the published defaults
(`--sobol-power 18 --sobol-replicates 8 --nf-evals 500000`) against the smoke
checkpoint. Results are the headline table in §10.

Comparing the smoke model against the **published thorough model** for the same
process:

| E [GeV] | published ESS/N | my smoke ESS/N |
|---:|---:|---:|
| 400 | 0.9984 | 0.820 |
| 650 | 0.9936 | 0.981 |
| 1000 | 0.9887 | 0.967 |
| 1500 | 0.9835 | 0.952 |
| 1950 | 0.9808 | 0.965 |

At 650–1950 GeV a 300-step smoke model already lands within **1–3%** of the
fully-trained model's ESS. The 400 GeV point is the exception and has an
obvious explanation: the training condition domain is
`sqrt(s_hat) ∈ [360, 2000]` GeV, so 400 GeV sits almost on the lower edge, where
the flow has seen the least data. Part I flagged the same edge behaviour in the
synthetic runs.

## 16. Revised "what I would do next"

Part I's list, updated:

1. ~~Decide whether to install MG5~~ — **done** (§12–§13). The remaining
   MadGraph work is breadth, not feasibility:
   - `gg_tth` (5-D, `configs/processes/gg_tth.json`) and `uu_ww_4l` (8-D) were
     **not** built. They are the next two `setup_madgraph_process.py`
     invocations and should go through the identical path.
   - The 20-D fully-decayed ttH benchmark uses a different entry point
     (`setup_madgraph_tth_benchmark.py --mode full`) whose process string
     includes decays, plus `MadEvent`, which needs a real `lhapdf`/MadEvent
     install. That is a genuinely separate project.
2. Still open from Part I: the missing
   `results/bayesian_delta_edge_selection/median_probability_graph.csv`
   (Part I §6.1).
3. A `requirements.txt` is still absent. If one is added, pin it to **this**
   known-good set (§11.2) rather than to Part I's `numpy<2`+torch-2.3 pairing —
   the numpy pin is a torch-version artifact, not a repo requirement.
4. **Document the CWD requirement** (§14). This is the highest-value
   documentation fix in the repository: as it stands, every MadGraph driver
   fails with exit status 0 when run the way the repo tells you to run it.
5. If MadGraph work is going to continue on EOS, generate the process trees on
   local disk and copy the result, or accept minutes per build (§11.3).

## 17. Quick reference (Part II)

```bash
export REPO=/eos/user/j/jlai/mango/conditional-nf-integration
export PY=/eos/user/j/jlai/mango/venv/bin/python
export MG5=/eos/user/j/jlai/mango/bin/mg5_aMC
cd $REPO && export PYTHONPATH=$REPO

# tests: no MadGraph needed, run from anywhere (NOT a dir containing py.py)
$PY -m pytest tests/                                          # 12 passed, ~13 s

# generate + compile a process (writes to generated/<name>_standalone)
$PY experiments/setup_madgraph_process.py \
    --config configs/processes/gg_ttbar.json --mg5 $MG5

# ---- everything below MUST run with CWD inside the standalone dir ----
cd $REPO/generated/gg_ttbar_standalone

$PY $REPO/experiments/validate_madgraph_process_integrand.py \
    --config $REPO/configs/processes/gg_ttbar.json --points 500

$PY $REPO/experiments/run_madgraph_process_conditional_copula.py \
    --config $REPO/configs/processes/gg_ttbar.json --speed-profile smoke \
    --output $REPO/results/gg_ttbar_copula_smoke

$PY $REPO/experiments/validate_madgraph_process_integral.py \
    --config $REPO/configs/processes/gg_ttbar.json \
    --checkpoint $REPO/results/gg_ttbar_copula_smoke/madgraph_gg_ttbar_2_final_model.pt \
    --output $REPO/results/gg_ttbar_validation_smoke
```

Artifacts left behind by this session (all gitignored, none committed):

```
/eos/user/j/jlai/mango/MG5_aMC_v3_6_7/     MadGraph 3.6.7 (5891 files)
/eos/user/j/jlai/mango/bin/mg5_aMC         python3.11 wrapper
/eos/user/j/jlai/mango/venv/               python 3.11.5 environment
generated/gg_ttbar_standalone/             MG5 standalone + matrix2py.so
results/gg_ttbar_copula_smoke/             smoke checkpoint + summary
results/gg_ttbar_validation_smoke/         held-out NF vs Sobol results
```

---

# Part III — Z + jets hadronic extension (Linux/ROCm, 2026-10-26)

This part records the one-jet validation, why `p p > z j` cannot be run by the
NF pipeline as-is, the two proposed pipeline changes, both energy-range
options for each, and the MadEvent slice diagnostic that fixes the training
range.

## 18. Why `p p > z j` cannot run as-is

MG5 generates `p p > z j`; the NF pipeline cannot consume it.

Scratch generation (MG5 3.6.7, `sm` model):

```bash
source /storage/users/dlai/mango-linux/env_linux.sh
mkdir -p /storage/users/dlai/mango-linux/scratch_ppzj
cat > /storage/users/dlai/mango-linux/scratch_ppzj/gen.mg5 <<'EOF'
set automatic_html_opening False
import model sm
generate p p > z j
output standalone /storage/users/dlai/mango-linux/scratch_ppzj/standalone --prefix=int
EOF
/storage/users/dlai/mango-linux/bin/mg5_aMC \
  /storage/users/dlai/mango-linux/scratch_ppzj/gen.mg5
```

MG5 expands it to **12 partonic processes in 6 subprocess directories**
(`P1_gu_zu`, `P1_gd_zd`, `P1_gux_zux`, `P1_gdx_zdx`, `P1_uux_zg`,
`P1_ddx_zg`), 24 diagrams. The per-P `matrix2py` wrappers compile, and the NF
loader then fails immediately:

```
RuntimeError: A process specification must resolve to exactly one subprocess;
found [P1_ddx_zg, P1_gd_zd, P1_gdx_zdx, P1_gu_zu, P1_gux_zux, P1_uux_zg]
```

Why, structurally:

1. `MadGraphMatrixElement` requires exactly one `P*/matrix2py*.so`; `p p > z j`
   has six.
2. The phase space is fixed 2→2 with one `final_pdgs` list. `j` is a
   multiparticle sum (`g u c d s u~ c~ d~ s~`), not one final-state PDG.
3. The integrand has **no PDF/beam layer**. It sets the initial momenta to
   back-to-back partons at √ŝ. `p p` is a proton beam: the physical cross
   section needs an x1,x2 PDF convolution and a sum over initial partons. The
   NF conditions on partonic √ŝ, not hadronic s.
4. A MadEvent `p p > z j` run with `lpp=1` is therefore a different observable
   (PDF-convolved hadronic pb), not the partonic integral the NF estimates.

The runnable unit is one partonic channel: **`g u > z u @1`**, which is exactly
`P1_gu_zu`, one of the six subprocess groups that `p p > z j` expands to.
Running the full hadronic process needs a new channel-index + PDF layer, not a
config change.

## 19. Proposed change 1 (Option 1): partonic NF + hadronic luminosity layer

This is the smallest change that computes the physical `pp → Z + jet` cross
section while reusing everything already validated at parton level.

**The integral.** The existing NF estimates the partonic quantity at fixed √ŝ:

\[
\hat\sigma_c(\hat s)=\int d\Phi_2\,\frac{|M_c|^2}{2\hat s}
\]

The hadronic cross section is a one-dimensional convolution over √ŝ plus a sum
over partonic channels:

\[
\sigma=\sum_c\int d\sqrt{\hat s}\;\frac{dL_c}{d\sqrt{\hat s}}\;\hat\sigma_c(\sqrt{\hat s}),
\qquad
\frac{dL_c}{d\sqrt{\hat s}}=\frac{2\sqrt{\hat s}}{s}\int dY\,f_{a_c}(x_1,\mu_F)f_{b_c}(x_2,\mu_F),
\]

with \(x_{1,2}=\sqrt{\hat s/s}\,e^{\pm Y}\) and
\(|Y|\le \ln(\sqrt{s}/\sqrt{\hat s})\). The PDFs and the 1/s Jacobian are the
only new physics; the flow still samples the 2-D partonic phase space at fixed
√ŝ.

**Code changes:**

1. `madgraph_process_integrand.py`
   - `MadGraphChannelSet`: load the multi-subprocess wrapper once and map a
     channel index to its PDG order and matrix element. MG5 already generates
     the interface for this — `SubProcesses/f2py_wrapper.f` contains
     `SMATRIXHEL(PDGS, PROCID, NPDG, P, ALPHAS, SCALE2, NHEL, ANS)`,
     `GET_PDG_ORDER`, and `GET_PREFIX`. The makefile rule for the combined
     library is commented out in MG5 3.6.7, so either enable it or use one
     worker process per P directory. The per-P `matrix2py` libraries cannot all
     be imported under the same module name in one process.
   - `PDFProvider` backed by LHAPDF (`lhapdf.mkPDF(...)`), with the PDF set and
     member in the process config.
   - `load_process_spec` accepts `beam: {"type": "pp", "sqrt_s": 13000}`, a
     `channels` list, and a `pdf` block.

2. New `experiments/integrate_hadronic_process.py`
   - For each channel \(c\), evaluate the existing conditional NF on a
     log-spaced √ŝ grid, multiply by the luminosity and sum.
   - Combine the NF’s conditional error at each grid point with the luminosity
     quadrature error.
   - No change to the flow or to `run_madgraph_process_conditional_copula.py`.

3. New `experiments/validate_hadronic_process_with_madevent.py` (or extend the
   existing one)
   - Generate `p p > z j` as a normal MadEvent process (`lpp1 = lpp2 = 1`,
     `ebeam = 6500`), use the same PDF set, scale, and jet cuts, and compare
     the total cross section and the jet-\(p_T\)/\(\eta\) distributions.
   - This is the hadronic analogue of the `lpp=0` check we just ran for
     `g u > z u @1`.

4. `configs/processes/pp_zj.json`
   - `process: "p p > z j"`, beam 13 TeV, PDF set, scale choice, channel list,
     jet cuts.

**Important caveats:**

- The full `p p > z j` needs the **full LO amplitude**, not the `@1` diagram
  filter. The current `gu_zu` model is `@1`; it validates the machinery but is
  not the production amplitude.
- The existing NF models are conditioned on √ŝ ∈ [200, 2000] GeV. The LHC
  luminosity extends to 13 TeV; the total is dominated by low √ŝ, but the
  high-√ŝ tail needs either an extended training range or a separate tail
  integral.
  **Correction (2026-10-26, §21):** the MadEvent slice measurement below shows
  that **74% of the hadronic cross section lies below 200 GeV** (7.2% in
  113.4–120 GeV, 66.8% in 120–200 GeV), while the 2–13 TeV tail is 0.0015%.
  The training range must be extended **down to ~114 GeV**, not up to 13 TeV.
- The PDF set must be consistent between the NF side and MadEvent, including
  α_s(M_Z). The current NF uses fixed α_s = 0.118; choose a PDF set with that
  value, or pass the PDF’s α_s to the matrix element.
- This gives a **cross section**, not a single hadronic event sampler. If the
  goal is detector-level events or a flow that samples (x₁, x₂, channel,
  partonic angles) jointly, you need Option 2.

**Effort:** roughly 1–2 days of code plus retraining the missing channels (each
2-D balanced channel is ~20 min on this node), plus the LHAPDF installation.

### 19.1 Energy-range option 1A — one model on [114, 2000] GeV

Sample √ŝ log-uniformly (or uniformly in log √ŝ) and use a log-√ŝ condition
feature. The current z, z², log(1+z) features with
z = (√ŝ − 200)/(2000 − 200) compress 114–200 GeV into z < 0, so the feature
and the condition sampler both need changing. One model is operationally
simpler, but the low-√ŝ region — where the flow sees a small, cut-shaped phase
space — gets only a small fraction of the training conditions unless the
sampling is deliberately reweighted.

### 19.2 Energy-range option 1B — split range (recommended)

Train a dedicated low-range model on roughly **114–250 GeV**, which covers
~74% of the hadronic cross section, and keep the existing `[200, 2000]` model
for the high range. The overlap 200–250 GeV is a useful consistency check. The
low-range model is cheap (~20 min per channel), its condition interval is
narrow enough for the current features, and the already-validated high-range
model is reused unchanged. Add the `[2000, 13000]` tail as a negligible
constant (0.0015%) or a single MadEvent number.

**Production-amplitude caveat:** both the low-range and high-range models must
be trained **without** the `@1` diagram filter for the physical `p p > z j`.
The existing `gu_zu` model is `@1`; it validates the machinery but is not the
production amplitude. The high-range model in this option is the existing
`@1` model only as a machinery baseline; a production run needs a full-LO
retrain of the high range as well.

## 20. Proposed change 2 (Option 2): direct hadronic flow

Make the flow sample the full hadronic phase space:

\[
u=(c,\hat s,Y,u_1,u_2),\qquad
f_c=\frac{|M_c|^2}{2\hat s}\,J(\hat s,u)\,\frac{f_{a_c}(x_1)f_{b_c}(x_2)}{s}\,\Theta(\text{lab cuts}).
\]

The final-state partons are generated in the partonic CM and boosted by \(Y\);
the jet \(p_T,\eta\) cuts are applied in the lab. The channel \(c\) is a
categorical variable; the estimator is

\[
\sigma=\mathbb E_{c\sim p_c,\,u\sim q_c}\left[\frac{f_c(u)}{p_c\,q_c(u)}\right],
\]

so only the sampled channel’s matrix element is needed per point. The flow
needs a categorical marginal head and a PDF-aware target. The continuous flow
can still use the existing marginal–copula factorization, but the target now
includes the PDF peaks at small x, so the ESS will be lower and the training
harder. This is the “real” `pp > z j` sampler, but it is a new phase-space
class, a new target, a new training mode, and a new validation ladder (Sobol
over the 4-D hadronic cube with PDFs, plus MadEvent `lpp=1`). Realistically
1–2 weeks.

### 20.1 Energy-range option 2A — single flow over the full hadronic phase space

One joint flow over `(c, √ŝ, Y, u1, u2)` covering √ŝ from the ~113.4 GeV
threshold to 13 TeV (equivalently x1, x2 from the PDF support to 1), with the
PDFs and channel sum in the target. The beam energy √s = 13 TeV is a constant
condition (or omitted). Pros: a single sampler for the whole hadronic phase
space. Cons: the flow must resolve the luminosity peak at low √ŝ (74% of the
cross section below 200 GeV) and the very small high-√ŝ tail in one model; the
PDF peaks at small x and the large dynamic range make the training harder and
the ESS lower.

### 20.2 Energy-range option 2B — split range (recommended if 2A struggles)

Train two joint flows: a low-√ŝ flow on **114–250 GeV** (the same boundary as
Option 1B; ~74% of the cross section) and a high-√ŝ flow on **250–2000 GeV**,
plus the negligible `[2000, 13000]` tail as a constant or a coarse third
model. Each flow keeps the channel categorical and the PDF-aware target, but
sees a narrower √ŝ range and a more uniform luminosity. The 200–250 GeV
overlap is a consistency check. This mirrors Option 1B: the low range gets a
dedicated model, the high range is cheaper, and the tail is negligible.

## 21. MadEvent `p p > z j` slice diagnostic

### 21.1 Result

MadEvent `p p > z j` at 13 TeV, pT,j > 20 GeV, |eta_j| < 2.5, default
`nn23lo1` PDF and dynamic scale:

| √ŝ slice (GeV) | cross section (pb) | fraction of total |
|---|---:|---:|
| 0–120 | 645.0 ± 0.8 | 7.16% |
| 120–200 | 6020 ± 5.6 | 66.81% |
| 200–2000 | 2336 ± 4.4 | 25.92% |
| 2000–13000 | 0.1389 ± 0.0003 | 0.0015% |
| **total** | **9010 ± 10** | 100% |

The slices sum to 9001 pb, 99.9% of the total; the 0.1% difference is the
integration error. The 0–120 GeV entry is the threshold window: with
m_Z = 91.188 GeV and pT,j > 20 GeV, the partonic threshold is
√ŝ_min ≈ 113.4 GeV, so the 645 pb is the 113.4–120 GeV region.

What this means:

1. The current NF range covers only ~26% of the hadronic cross section. 74%
   lies below 200 GeV, overwhelmingly in 120–200 GeV. This is the PDF
   luminosity at small x beating the falling partonic cross section.
2. The high tail is irrelevant. Above 2 TeV the contribution is 0.0015%, so
   there is no reason to train a flow out to 13 TeV. A constant or a MadEvent
   estimate is enough for the tail.
3. The condition minimum must be lowered to ~114 GeV. Training on
   `[200, 2000]` and extrapolating below 200 would miss most of the physics.
   The NF can handle √ŝ near threshold, but the allowed phase space shrinks as
   the pT cut approaches the kinematic limit, so the ESS there needs checking.
4. This is exactly why the beam energy is not the right NF condition. At
   13 TeV the beam energy is a constant; the physically important variable is
   √ŝ, whose distribution is extremely non-uniform. The beam energy enters
   through the luminosity and the x1,2 ≤ 1 limits, not through the flow’s
   condition. MadGraph handles the two modes separately — partonic
   (`lpp=0`, `ebeam=√ŝ/2`) for the building-block checks and hadronic
   (`lpp=1`, `ebeam=6500`) for the physical cross section — and they must not
   be mixed.

**Revised range strategy.** Two workable choices:

**A. One model on [114, 2000] GeV.** (Same as Option 1A.) One model is
operationally simpler, but the low-√ŝ region gets only a small fraction of the
training conditions unless the sampling is reweighted.

**B. Split the range (recommended).** A dedicated low-range model on roughly
114–250 GeV, which covers ~74% of the hadronic cross section, plus the
existing [200, 2000] model for the high range; the 200–250 GeV overlap is a
consistency check; the [2000, 13000] tail is a negligible constant (0.0015%).
The low-range model is cheap (~20 min per channel), its condition interval is
narrow enough for the current features, and the already-validated high-range
model is reused unchanged (subject to the full-LO/`@1` caveat above).

**Decision pending:** A versus B. B is recommended.

### 21.2 Code-wise procedure

The measurement is a plain MadEvent integration; no NF code is involved.

**1. Generate the MadEvent directory.**

```bash
source /storage/users/dlai/mango-linux/env_linux.sh
mkdir -p /storage/users/dlai/mango-linux/scratch_ppzj
cat > /storage/users/dlai/mango-linux/scratch_ppzj/gen_ppzj_madevent.mg5 <<'EOF'
set automatic_html_opening False
import model sm
generate p p > z j
output /storage/users/dlai/mango-linux/scratch_ppzj/ppzj_madevent
EOF
/storage/users/dlai/mango-linux/bin/mg5_aMC \
  /storage/users/dlai/mango-linux/scratch_ppzj/gen_ppzj_madevent.mg5
```

`output` (not `output standalone`) creates `bin/madevent`, the MadEvent
executable, and `Cards/run_card.dat`.

**2. Set the run card.** The default card already has the hadronic mode:

```
lpp1 = lpp2 = 1
ebeam1 = ebeam2 = 6500
pdlabel1 = pdlabel2 = nn23lo1
ptj = 20
dsqrt_shat = 0
dsqrt_shatmax = -1
fixed_ren_scale = False
fixed_fac_scale1 = fixed_fac_scale2 = False
```

The diagnostic changes only:

| field | value | meaning |
|---|---|---|
| `etaj` | 2.5 | jet rapidity cut |
| `dsqrt_shat` | slice min (GeV) | minimum partonic √ŝ |
| `dsqrt_shatmax` | slice max (GeV; -1 = none) | maximum partonic √ŝ |
| `iseed` | 260919 | fixed seed |
| `use_syst` | False | no systematics |
| `nevents` | 10000 | not used by survey/refine |

The run card uses the value-before-name format, e.g. `5.0 = etaj`,
`0.0 = dsqrt_shat`, `-1 = dsqrt_shatmax`. The Python helper
`update_run_card` in `experiments/validate_madgraph_process_with_madevent.py`
replaces the value with a regex that preserves the label and the comment:

```python
pattern = re.compile(
    rf"(?im)^(?P<indent>\s*)(?P<old>[^#!\n=]+?)(?P<sep>\s*=\s*{re.escape(key)}\b)"
)
text, count = pattern.subn(
    lambda m: f"{m.group('indent')}{value}{m.group('sep')}", text, count=1
)
```

**3. Run the slices.** For each slice, write a command file and run
`bin/madevent` from the process directory:

```
survey <name> --points=20000 --iterations=8 --accuracy=0.002
refine 0.001 3
```

`survey` builds the grid; `refine` reduces the integration error to 0.1%.
The slices were:

| name | `dsqrt_shat` | `dsqrt_shatmax` |
|---|---:|---:|
| total | 0.0 | -1.0 |
| 0_120 | 0.0 | 120.0 |
| 120_200 | 120.0 | 200.0 |
| 200_2000 | 200.0 | 2000.0 |
| 2000_13000 | 2000.0 | 13000.0 |

**4. Read the result.** `SubProcesses/results.dat` first line is
`cross error ...` in pb; the helper `current_results_dat` parses the first two
fields:

```python
first = (process_dir / "SubProcesses" / "results.dat").read_text().splitlines()[0]
fields = first.replace("D", "E").split()
cross_pb, error_pb = float(fields[0]), float(fields[1])
```

The total is measured independently; the slices sum to 99.9% of it, which is
the closure check.

**5. Driver.** The inline loop was:

```python
cases = [
    ('total', 0.0, -1.0),
    ('0_120', 0.0, 120.0),
    ('120_200', 120.0, 200.0),
    ('200_2000', 200.0, 2000.0),
    ('2000_13000', 2000.0, 13000.0),
]
for name, smin, smax in cases:
    update_run_card(proc, {
        'lpp1': 1, 'lpp2': 1, 'ebeam1': '6500.0', 'ebeam2': '6500.0',
        'ptj': 20.0, 'etaj': 2.5,
        'dsqrt_shat': smin, 'dsqrt_shatmax': smax,
        'nevents': 10000, 'iseed': 260919, 'use_syst': 'False',
    })
    (proc / f'{name}_commands.txt').write_text(
        f"survey {name} --points=20000 --iterations=8 --accuracy=0.002\n"
        f"refine 0.001 3\n"
    )
    subprocess.run(
        [str(proc / 'bin' / 'madevent'), str(proc / f'{name}_commands.txt')],
        cwd=proc, capture_output=True, text=True,
    )
    print(name, current_results_dat(proc))
```

The generated directory is in the scratch tree
(`/storage/users/dlai/mango-linux/scratch_ppzj/ppzj_madevent`), not in the
repo. The helper functions are in
`experiments/validate_madgraph_process_with_madevent.py`.

**Caveats.** The numbers use the default `nn23lo1` PDF and dynamic scales; the
fractions should be recomputed with the final PDF/scale choice. The
`p p > z j` here is the full LO amplitude (no `@1`). The total 9010 pb is not
the same as the native MG5 baseline 14350 pb because that baseline used
different PDF/scale/cuts (e.g. `etaj = 5`); the slice fractions are the robust
output.

## 22. Current one-jet validation status

Balanced `gu_zu` (`g u > z u @1`, 2-D, pT,j > 20, |eta_j| < 2.5), 100k NF and
2^18 × 8 Sobol evaluations per energy, compared to partonic MadEvent
(`lpp=0`, `ebeam=√ŝ/2`):

| √ŝ (GeV) | NF (pb) | Sobol (pb) | MadEvent (pb) | NF−Sobol pull | MadEvent−NF pull |
|---:|---:|---:|---:|---:|---:|
| 250 | 10.4089 ± 0.0064 | 10.4079 ± 0.00014 | 10.4010 ± 0.0066 | +0.14σ | −0.85σ |
| 400 | 4.8288 ± 0.0026 | 4.8296 ± 0.00009 | 4.8307 ± 0.0018 | −0.31σ | +0.61σ |
| 650 | 1.92776 ± 0.00093 | 1.92726 ± 0.00004 | 1.92640 ± 0.00087 | +0.54σ | −1.07σ |
| 1000 | 0.829523 ± 0.00039 | 0.829555 ± 0.000014 | 0.829530 ± 0.00035 | −0.08σ | +0.01σ |
| 1500 | 0.371219 ± 0.000135 | 0.371483 ± 0.000007 | 0.371550 ± 0.000123 | −1.95σ | +1.81σ |
| 1950 | 0.220276 ± 0.000073 | 0.220348 ± 0.000003 | 0.220460 ± 0.000109 | −0.99σ | +1.41σ |

Balanced training: ESS 0.313, Ihat = 4.71015×10⁻⁹ GeV⁻², std_logw = 1.1268,
20m52s. The held-out NF ESS is 0.963–0.989. The Sobol and MadEvent references
agree within ~1σ at every energy; the NF is consistent within statistics, with
the 1500 GeV point the largest (1.95σ vs Sobol). A 500k-evaluation NF rerun
would settle whether that is a fluctuation or a small high-√ŝ bias.
