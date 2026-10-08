# Introduction to `conditional-nf-integration`

*An orientation to the repository: what it does, how it is built, what its
results claim, and what you can actually run.*

---

## 1. The idea in one paragraph

Almost every multidimensional physics calculation needs a Monte Carlo integral
of the form

$$
I(c)=\int_{[0,1]^d} f(x\mid c)\,dx ,
$$

where `c` is some external parameter — here, the partonic centre-of-mass energy
$\sqrt{\hat s}$. The classical approach, VEGAS, adapts its sampling grid
separately for every value of `c` you care about. This project instead trains
**one conditional normalizing flow** across a continuous range of `c`, and then
reuses it at held-out values of `c` with no retraining.

The important design choice is what the network is allowed to do. It does
**not** predict the integral, and it does **not** replace the integrand. It only
learns *where to put the samples*. The final answer is still an ordinary
importance-sampling average:

$$
\widehat I_N(c)=\frac1N\sum_{n=1}^N \frac{f(x_n\mid c)}{r_\theta(x_n\mid c)},
\qquad x_n\sim r_\theta(\cdot\mid c).
$$

This separation is deliberate: a bad proposal costs you **variance**, never
**correctness**. The exact integrand stays in the estimator.

---

## 2. The method

### 2.1 Marginal–copula factorization

The proposal is not a single unstructured density on $[0,1]^d$. It is split
into two pieces:

$$
q_\theta(x\mid c)=c_\theta(u_1,\ldots,u_d\mid c)\prod_{i=1}^d q_{i,\theta}(x_i\mid c),
\qquad u_i=F_{i,\theta}(x_i\mid c)
$$

- **Conditional marginals** $q_{i,\theta}$ learn how each coordinate's shape
  changes with `c`. Each is a **monotone cubic-Hermite spline** on $[0,1]$: a
  small network maps the condition to positive bin widths and heights, and
  PCHIP-style knot derivatives keep the map monotone. The inverse is bracketed
  per-bin bisection plus one Newton correction.
- **The autoregressive copula** $c_\theta$ then models whatever dependence
  remains *after* each coordinate has been pushed through its own marginal CDF
  to a rank coordinate $u_i\in[0,1]$.

The motivation is separation of concerns: a narrow or lopsided coordinate is
handled by its own 1-D marginal, while curved or multimodal correlations are
handled by the copula in rank space. Fitting the marginals first also gives the
copula a much better-conditioned problem.

The scalar condition is normalized to $z\in[0,1]$ and expanded into three
features — $(z,\ z^2,\ \log(1+z)/\log 2)$ — rather than fed in raw.

### 2.2 Defensive mixture

The learned density is always mixed with a uniform proposal:

$$
r_\theta(x\mid c)=(1-\varepsilon)\,q_\theta(x\mid c)+\varepsilon .
$$

On the unit hypercube this guarantees full support. If the flow has not yet
discovered some region of the integrand, the uniform component still samples it,
so the estimate degrades gracefully instead of silently missing mass.

### 2.3 The diagnostic that matters

Every result is reported with **normalized effective sample size**:

$$
\frac{\mathrm{ESS}}{N}=\frac{(\sum_n w_n)^2}{N\sum_n w_n^2},
\qquad w_n=\frac{f(x_n\mid c)}{r_\theta(x_n\mid c)} .
$$

ESS/N near 1 means the importance weights are nearly constant, i.e. the proposal
is close to the ideal $f/\int f$. ESS/N near 0 means a few samples dominate the
estimate and the reported error bars are not trustworthy. This is the single
number to look at when judging a run.

### 2.4 Training without target samples

Notably, training needs **only evaluations of the unnormalized, nonnegative
integrand**. It does not need exact target samples and it does not need to know
the value of $I(c)$ — which is the whole point, since that value is what you are
trying to compute.

The documented workflow:

1. sample conditions across the training interval;
2. build self-normalized importance-weighted approximations to the target;
3. learn the conditional marginals;
4. transform to marginal-rank coordinates and learn the copula;
5. refresh the weighted cache with the improved proposal;
6. refine on the full target with importance-weighted KL training;
7. optionally fine-tune with a $\chi^2$/weight-variance objective, **retained
   only if held-out ESS improves**.

Annealed intermediate targets bridge the gap from a broad initial proposal to
the sharp final integrand. Reported integral estimates always use *fresh*
samples, not the ones the model was optimized on.

---

## 3. Repository map

```
nf/                             flow implementations
  conditional_cubic_1d.py         monotone cubic-Hermite conditional flow (the live core)
  conditional_linear_1d.py        its base class
  conditional_cubic_2d.py         2-D copula coupling
  ... (see §7 — most of this package is unused by the current experiments)

madgraph_process_integrand.py   configurable matrix element + 2/5/8-D phase space
madgraph_tth_integrand.py       resonance-aware 20-D ttH phase space

experiments/                    all drivers
configs/processes/              process definitions (gg_ttbar, gg_tth, uu_ww_4l)
tests/                          12 tests: spline correctness + phase-space validation
docs/                           methods, validation record, reproducibility guide
results_public/                 curated published results (CSV / JSON / PNG)
scripts/curate_public_results.py
```

### The architectural fact that isn't documented anywhere

**`experiments/run_adaptive_integrand_structure_search.py` (≈6,300 lines) is the
real engine.** It contains the marginal fitting, copula training, annealing,
Bayesian edge pruning and $\chi^2$ refinement.

The MadGraph physics drivers are thin adapters over it. For example,
`run_madgraph_process_conditional_copula.py` monkey-patches the engine's module
globals and then calls `baseline.main()`:

```python
baseline.SUPPORTED_BENCHMARKS = {benchmark}
baseline.BENCHMARK_DIMENSIONS = {benchmark: evaluator.dimension}
baseline.sample_conditions    = sample_conditions
baseline.target_log_integrand = target_log_integrand   # wraps MadGraph
baseline.main()
```

So "train on the 20-D ttH process" and "train on the 32-D synthetic benchmark"
are **the same code path with the integrand swapped out**. If you understand the
adaptive-search driver, you understand the whole project.

---

## 4. Two families of benchmark

### 4.1 Synthetic (no external dependencies)

Targets on $[0,1]^d$ with **known** integrals, used to test the method itself:

| name | character |
|---|---|
| `single_gaussian` | a simple truncated Gaussian |
| `sharp_broad` | narrow plus broad mixture |
| `curved_ridge` | curved, correlated support |
| `xshape` | X-shaped dependence structure |
| `regime_switch` | changes behaviour across the condition domain |
| `sparse_wave_32` | 32-D; condition-dependent autoregressive structure |

`sparse_wave_32` is the headline stress test: 32 dimensions, a known unit
integral, and strongly dependent conditional structure. The exact sampler exists
but is hidden from training and used only for retrospective diagnostics.

### 4.2 Scattering processes (MadGraph)

`c = sqrt(s_hat)` and the integrand is a generated matrix element times a
process-specific phase-space Jacobian:

$$
f(x\mid\sqrt{\hat s})=\frac{|\mathcal M(p(x))|^2}{2\hat s}\,J(x).
$$

| process | dim | final state |
|---|---:|---|
| $gg\to t\bar t$ | 2 | massive two-body control |
| $gg\to t\bar tH$ | 5 | massive three-body |
| $u\bar u\to W^+W^-\to e^+\nu_e\mu^-\bar\nu_\mu$ | 8 | doubly resonant four-body |
| fully decayed $gg\to t\bar tH$ | 20 | resonance-aware, five Breit-Wigner masses + 14 angular variables |

Each process needs a `standalone` MadGraph export with a compiled
`matrix2py` library, built by `experiments/setup_madgraph_process.py` from the
JSON in `configs/processes/`.

---

## 5. What the results claim

### Scattering: conditional reuse works

One trained model, evaluated at held-out energies, compared against independent
randomized Sobol integration and (for 20-D) explicitly refined MadEvent:

| Process | Dim | Held-out ESS/N |
|---|---:|---:|
| $gg\to t\bar t$ | 2 | 0.981–0.998 |
| $gg\to t\bar tH$ | 5 | 0.959–0.987 |
| $u\bar u\to W^+W^-\to 4\ell$ | 8 | 0.927–0.941 |
| fully decayed $gg\to t\bar tH$ | 20 | 0.885–0.941 |

A provisional 2.30σ discrepancy at 300 GeV in the 8-D case was retested with
higher statistics and became a **0.18σ** difference:

$$
\sigma_{\rm NF}=0.0529874\pm0.000013\ \mathrm{pb},\qquad
\sigma_{\rm Sobol}=0.0529849\pm0.0000031\ \mathrm{pb}.
$$

### 32-D synthetic: the proposal beats axis-adaptive VEGAS

160 held-out conditions, ≈10⁶ integrand evaluations per method per condition:

| Method | Mean estimate | Mean absolute error | Mean ESS/N | 95% coverage |
|---|---:|---:|---:|---:|
| conditional NF | 0.999254 | 0.001367 | 0.5328 | **0.7875** |
| tested VEGAS configuration | 0.0000135 | 0.999987 | — | 0.0000 |

Read this carefully. The claim is **narrow**: on this particular correlated 32-D
target, with this particular axis-adaptive VEGAS setup and budget, the learned
autoregressive proposal represents the target and VEGAS does not. It is not a
claim that VEGAS fails in general.

And the NF's own coverage of **0.7875 against a nominal 0.95 is a real
undercoverage problem**, which the authors surface rather than hide — plausibly
rare importance-weight tails. Point estimates are accurate in aggregate; the
per-condition error bars are not calibrated.

---

## 6. Getting started

### Environment

There is **no `requirements.txt` and no `environment.yml`**; `pyproject.toml`
declares no dependencies at all. The real set, reverse-engineered from the
imports, is:

```
numpy  scipy  torch  pandas  matplotlib  pytest
vegas        # only for run_copula_vs_vegas_integrand.py
```

A set that has been verified end to end against this repository:

| package | verified version |
|---|---|
| python | 3.11.5 |
| numpy | 1.26.4 |
| torch | 2.14.1+cpu |
| scipy | 1.17.1 |
| pandas | 3.0.6 |
| matplotlib | 3.11.2 |
| vegas | 6.4.1 |
| pytest | 9.1.1 |

No GPU is required to run anything here, but note that the **published results
were produced on CUDA** — expect the larger benchmarks to be substantially
slower on CPU than the quoted timings suggest. The synthetic drivers and the
2-D scattering process have been run to completion on CPU.

**On `numpy<2`.** You will see `numpy<2` recommended in older notes about this
repository. That pin is **not a requirement of the code** — it is a requirement
of *torch 2.3.x specifically*, which was compiled against NumPy 1.x and
otherwise raises `RuntimeError: Numpy is not available` from every `.numpy()`
call. With `torch 2.14.1+cpu` it is unnecessary, and NumPy 2 works. Pin it only
if you are also pinning an old torch.

On a machine without a NumPy-compatible wheel for `vegas`, use
`pip install --only-binary :all: vegas`.

### Gotcha 1 — `sys.path` (all drivers)

Running a driver directly usually fails with:

```
ModuleNotFoundError: No module named 'nf.conditional_cubic_1d'; 'nf' is not a package
```

because Python puts the *script's* directory (`experiments/`) on `sys.path`, not
the repository root. Fix it once:

```bash
export PYTHONPATH="$PWD"     # from the repo root
# or: pip install -e .
```

Also note every driver writes to **relative** paths (`results/...`).
`results/` is gitignored; it does not exist in a fresh clone.

### Gotcha 2 — the working directory (MadGraph drivers only)

The usual advice is "always run from the repository root". **For every
MadGraph-dependent script, that advice is wrong and fails silently.**

MadGraph's generated `setpara`/`lha_read` resolves its auxiliary Les Houches
files relative to the **current working directory**, not relative to the
parameter card it is handed. It looks for `./Cards/ident_card.dat` and then
walks up five parent directories. If it does not find one, the Fortran code
executes a `stop` — and Fortran `stop` **exits with status 0**. You get:

```
 Warning: file ident_card.dat not found in the parent directories!(lha_read)
$ echo $?
0
```

No traceback, no error, no output from any print statement after the failing
call. A run that looks like it succeeded did nothing at all.

The fix is to run with the working directory inside the generated standalone
tree, where `Cards/` lives, and pass **absolute** paths for everything else:

```bash
cd generated/gg_ttbar_standalone          # CWD must see Cards/
PYTHONPATH=$REPO python $REPO/experiments/validate_madgraph_process_integrand.py \
    --config $REPO/configs/processes/gg_ttbar.json --points 500
```

Do not try to fix this with a single `Cards/` symlink in the repo root — every
process has its own `Cards/`, so such a link can only be correct for one
process at a time.

### Gotcha 3 — a stray `py.py` breaks `pytest`

After running the MadGraph setup step from a directory, that directory may
contain a new file called **`py.py`**. It is not yours and not part of the
repository: PLY, driven by MadGraph's ALOHA code generator, writes a generated
LALR parser table into the current working directory under that name. Since
Python puts the CWD first on `sys.path`, it shadows the `py` compatibility shim
that pytest ships, and every `pytest` run from that directory then dies with:

```
AttributeError: module 'py' has no attribute 'path'
```

Delete the stray `py.py` (MadGraph may also leave `nsqso_born.inc` and
`param.log` behind; both are harmless). Nothing in the repository depends on it.

### Fastest ways to confirm a working install

```bash
python -m pytest tests/                                    # 12 tests
python experiments/run_copula_bin_sweep.py --smoke         # ~1 min, writes plots
```

Both are self-contained and need no MadGraph. The second is the better end-to-end
check: it exercises torch, pandas, matplotlib and the `nf` package together and
produces a `summary.csv` plus eight PNGs.

### Longer runs

```bash
# ~2 min: NF vs VEGAS on a 2-D target (needs the vegas package)
python experiments/run_copula_vs_vegas_integrand.py --smoke

# ~15-25 min on CPU: the full engine, with its annealing schedule
python experiments/run_adaptive_integrand_structure_search.py \
    --speed-profile smoke --no-compile-kernels
```

The engine has a built-in compute dial, `--speed-profile {smoke,fast,balanced,thorough}`.
The default `thorough` uses `kl_max_steps = 30_000` against smoke's `300` — about
**100×** the budget. Always start with `--speed-profile smoke`.

### The MadGraph dependency — installed and exercised

Part of this repository runs with no external dependencies at all; part of it
cannot run without a MadGraph installation. The split is clean:

**Runs without MadGraph**

- `tests/` — including the phase-space validation (see below).
- Every synthetic driver: `run_copula_bin_sweep.py`,
  `run_copula_vs_vegas_integrand.py`,
  `run_adaptive_integrand_structure_search.py`,
  `run_bayesian_selected_copula_benchmark.py`,
  `run_iterative_bayesian_edge_pruning_frozen_null.py`.
- `madgraph_process_integrand.py` and `madgraph_tth_integrand.py` **import**
  fine — they are pure NumPy phase-space code. Only the
  `MadGraphMatrixElement` class, which loads a compiled `matrix2py*.so`, needs
  the generated library.

**Requires MadGraph**

- `setup_madgraph_process.py` / `setup_madgraph_tth_benchmark.py` (generate and
  compile the standalone export)
- `run_madgraph_process_conditional_copula.py` /
  `run_madgraph_tth_conditional_copula.py` (train on a real process)
- `validate_madgraph_*` and `benchmark_nf_vs_madevent_tth.py` (the Sobol and
  MadEvent cross-checks)

`MadGraphMatrixElement` globs for `SubProcesses/P*/matrix2py*.so` — a **Linux
shared object**, so this needs a real Linux toolchain (or WSL/MSYS2 with
`gfortran`), not just a `pip install`. Note also that the process configs point
at `generated/<process>_standalone`, a gitignored directory absent from a fresh
clone; you must generate it yourself.

**This does work, and the procedure is short.** On a Linux box with `gfortran`:

```bash
# 1. MadGraph 3.6.7 (any recent 3.x should do)
curl -sLO https://launchpad.net/mg5amcnlo/3.0/3.6.x/+download/MG5_aMC_v3.6.7.tar.gz
tar xzf MG5_aMC_v3.6.7.tar.gz

# 2. MG5's bin/mg5_aMC has a "#!/usr/bin/env python3" shebang; make sure that
#    python is >= 3.10 (3.6.7 has dropped 3.9). A one-line wrapper is enough.

# 3. generate + compile the standalone export
cd $REPO
PYTHONPATH=$PWD python experiments/setup_madgraph_process.py \
    --config configs/processes/gg_ttbar.json --mg5 /path/to/mg5_aMC
```

Notes on what you will see:

- The optional-tool warnings at MG5 startup (`fastjet-config`, `eMELA-config`,
  `lhapdf-config` not found) are **harmless** — none are needed for LO partonic
  matrix elements with fixed α_s and no PDFs.
- `compile_libraries()` first tries the combined `make allmatrix2py.so` target.
  On MG5 3.6.7 that target is unavailable, so the script falls back to per-process
  `make matrix2py.so` and prints `Combined allmatrix2py target is unavailable;
  trying per-process wrappers.` That fallback is the path that works, and it
  produces exactly the `P*/matrix2py*.so` layout the loader wants.
- Remember Gotcha 2 above: everything after this must run with the CWD inside
  `generated/<process>_standalone/`.

Once built, the `gg → t t̄` chain has been run end to end: the repo's 12 tests
pass, the integrand validates (four-momentum conservation to 2e-13, positive
Jacobian, ~47 000 matrix-element evaluations/second on one core), a smoke
training run completes, and held-out validation reproduces the **published
Sobol integrals in `results_public/gg_ttbar_validation/` exactly**, at all five
held-out energies, with a smoke-budget NF agreeing to within 0.7σ. See the
"Part II" section of `PRD.md` for the full record.

So the scattering pipeline **does run**, and the `gg → t t̄` row of §5 is
reproducible on CPU in a few minutes per stage. That is one of the four
processes, though: the 5-D and 8-D benchmarks were not built here, and the 20-D
fully-decayed ttH needs a working MadEvent, which is a separate installation
again.

### Two drivers not worth attempting casually

- `run_copula_benchmark_sweep.py` has **no command-line interface at all**. It
  ignores `argv` (so `--help` silently starts a full run) and its budgets are
  module-level constants you must edit by hand.
- `run_iterative_bayesian_edge_pruning_frozen_null.py --smoke` still runs three
  benchmarks × three seeds; `--smoke` silently overrides `--seeds`.

---

## 7. Things a new reader should know

**Most of `nf/` is dead code.** Only three files are reachable from any working
entry point:

| module | used by |
|---|---|
| `conditional_cubic_1d.py` | the synthetic drivers + tests |
| `conditional_linear_1d.py` | base class of the above, only |
| `conditional_cubic_2d.py` | `run_copula_benchmark_sweep.py`, only |

`train.py`, `evaluate.py`, `losses.py`, `conditional_rq_1d.py`,
`conditional_autoregressive_cubic_2d.py`, the `cube_*` flows, `logit_rotation.py`
and `flows/realnvp_nd.py` are unreferenced by any live path. Two of them are
traps: `losses.py` looks live because `train.py` imports it, but `train.py`
itself is never imported; `logit_rotation.py` is imported only by the dead
`cube_spline_flow.py`. To understand the method as it is actually run, read
`nf/conditional_cubic_1d.py` (92 lines) and the adaptive-search engine.

**`run_bayesian_selected_copula_benchmark.py` cannot run out of the box.** Its
default `--graph-file` points at
`results/bayesian_delta_edge_selection/median_probability_graph.csv`, which no
script in the repository creates, and it fails immediately with
`FileNotFoundError`. Pass `--graph-file` explicitly, or use one of the graph
families it generates internally.

**`reanalyze_nf_vs_madevent_errors.py` rewrites its `--results` directory in
place.** Pointing it at `results_public/` overwrites tracked, published CSVs.
Run it only against a copy.

---

## 8. Scope and honest limitations

Stated by the authors:

- **Training cost is excluded** from quoted online timing comparisons; the
  advantage claimed is post-training.
- Most results use **one trained checkpoint** with independent evaluation
  samples — not a systematic study over independently trained models.
- Calculations are **parton-level and leading order**, with specialized
  phase-space maps and no hadronic PDFs in the generic sweep.
- The formulation **assumes a nonnegative integrand**; signed subtraction terms
  at higher perturbative orders are not handled.
- The 32-D experiment shows **undercoverage** (0.7875 vs 0.95) despite accurate
  aggregate point estimates.

Worth adding, having examined the tree:

- The process sweep changes dimension, resonance structure and matrix-element
  complexity *simultaneously*, so it is not a controlled dimension-only study.
- The published 32-D benchmark in `results_public/nf_vs_vegas_160/` **has no
  producing script in this repository** — the method labels it uses (`nf_pure`)
  appear nowhere in the code. It can be inspected and cross-checked, but not
  regenerated from this tree. The same is true of several other curated
  `results_public/` runs, since the working `results/` tree was never committed.
- Not everything is unregenerable, though. `results_public/gg_ttbar_validation/`
  **is** regenerable, and doing so reproduces its Sobol column to every printed
  digit. Both facts are worth knowing before judging the curated results as a
  whole.

The authors are unusually explicit about these boundaries, which is a point in
the repository's favour.

---

## 9. Where to read next

| document | contents |
|---|---|
| `README.md` | the project's own summary and results tables |
| `docs/METHODS_AND_APPLICATIONS.md` | full mathematical and implementation description |
| `docs/VALIDATION_RESULTS.md` | the 20-D validation ladder, step by step |
| `docs/REPRODUCIBILITY.md` | the documented 9-step validation order |
| `PRD.md`, Part I | working notes (Windows): repository structure, verified synthetic smoke runs, blockers |
| `PRD.md`, Part II | working notes (Linux): MadGraph 3.6.7 installed, the working-directory bug, and the reproduced `gg → t t̄` scattering results |
