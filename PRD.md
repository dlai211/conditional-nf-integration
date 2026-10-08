# PRD.md — working notes on the `conditional-nf-integration` repo

**Personal log.** Written while probing someone else's repository to work out
its structure and how to actually run it. Everything below is a record of what
was tested on this machine, what worked, what failed, and what was too slow to
finish. Date: 2026-10-08.

---

## 0. Provenance

A public GitHub repo, **not** the local user's own work:

- remote: `https://github.com/XuZhaozhong-1/conditional-nf-integration.git`
- sole author: **Zhaozhong Xu** `<zhaozhong@…sdcc.bnl.gov>` (Brookhaven National
  Laboratory)
- commits: 2026-09-20 → 2026-09-25 (8 commits, branch `main`)
- working tree was clean on clone except for one uncommitted change
  (`experiments/run_adaptive_integrand_structure_search.py`, +5 lines adding a
  `sys.path` shim — not mine, left untouched)

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
