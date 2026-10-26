# Session handoff — `gg_zjj` balanced run (2026-10-26)

**Repo:** `C:/Users/laiji/test/conditional-nf-integration` (branch `main`;
previous HEAD `938f31f`).
**Read first:** this file, then `CLAUDE.md` (machine policy) and
`docs/HIGH_DIM_FINDINGS.md` (the separate high-dimensional investigation).

## TL;DR

1. A **balanced** training run for the new 5-D `gg_zjj` process is **still
   running**. Do not start a second one; it writes its checkpoint only at the
   end.
2. When it finishes, run the 200k `--no-sobol --compare-uniform` validation in
   §3 (about 2–3 minutes). That is the only remaining step from the current
   request.
3. All code and documentation are committed. The live checkpoint and balanced
   result tree are local-only (`*.pt` is gitignored).

## 0. Environment

Windows 11, Git Bash. Use the `workspace` conda interpreter, not base Anaconda
(base has a broken CPU-only torch):

```bash
REPO=C:/Users/laiji/test/conditional-nf-integration
PY=C:/Users/laiji/.conda/envs/workspace/python.exe
```

Torch is `2.3.0+cu121` on an RTX 3060. The earlier GPU proof is
`scripts/verify_torch_gpu.py` and `RUNBOOK_GPU.md`.

Commands that load the MadGraph standalone must run **from
`$REPO/generated/gg_zjj_standalone`**: the Fortran library resolves
`Cards/ident_card.dat` relative to the CWD. The training and validation
commands below were run there.

## 1. Live balanced training

```bash
cd "$REPO/generated/gg_zjj_standalone"
PYTHONPATH="$REPO:$REPO/experiments" "$PY" -u \
  "$REPO/experiments/run_madgraph_process_conditional_copula.py" \
  --config "$REPO/configs/processes/gg_zjj.json" \
  --speed-profile balanced --device cuda \
  --batch-size 30000 --is-points 128 --me-chunk 512 \
  --output "$REPO/results/gg_zjj_copula_balanced_gpu"
```

- Windows PID **35776** (Git Bash PID 633), started **03:33**.
- Log: `/tmp/gg_zjj_balanced.log` =
  `C:\Users\laiji\AppData\Local\Temp\gg_zjj_balanced.log`.
- At **05:19** it was in KL round 0, **step 2400 / 12000**, `lr=3.0e-3`,
  `val NLL=-0.953735`, source ESS `0.5895`.
- Rough ETA: ~1 step/s; two adaptive KL rounds (cap 12,000 each), a χ² stage
  (cap 3,600) and cache rebuilds. Expect roughly **3–6 more hours** from 05:19,
  possibly up to ~8 h if both KL rounds hit their cap. Early convergence can
  shorten it: it requires `lr <= 1e-4` and three checks with
  `|relative improvement| < 1%`.

Check progress with:

```bash
tail -20 /tmp/gg_zjj_balanced.log
ps -W | grep python          # WINPID 35776
```

Look for `CHI^2`, `FINAL RESULT`, or a final summary. **Do not assume the
checkpoint exists just because the log is active** — the wrapper writes
`madgraph_gg_zjj_5_final_model.pt` and
`madgraph_gg_zjj_5_final_summary.csv` at the end. If the process is gone and the
log stops without `FINAL RESULT`, the run died and must be restarted from
scratch (there is no intermediate checkpoint).

### ESS progression observed so far

| stage | mean conditional ESS | mean raw ESS |
|---|---:|---:|
| β = 0.25 | 0.856 | 438.4 |
| β = 0.50 | 0.689 | 353.0 |
| β = 0.75 | 0.571 | 292.3 |
| β = 1.00 | 0.466 | 238.6 |
| post-anneal bridge | 0.437 | — |
| full-graph KL cache | 0.595 | 76.1 |

The bridge estimate was `4.397e-10 GeV^-2`. This is healthy; the old smoke run
ended at ESS 0.493.

**Task 1 acceptance check:** after the run, confirm `std_logw` is finite in
`results/gg_zjj_copula_balanced_gpu/madgraph_gg_zjj_5_final_summary.csv`.
`evaluate_ess` was fixed to compute the standard deviation over finite weights
only; cut-rejected `-inf` weights previously produced `NaN`. The smoke summary
has a **blank** `std_logw` because it predates that fix — do not treat it as a
regression.

## 2. What is already done

### Process and integrand

- `configs/processes/gg_zjj.json`: `g g > z u u~ @1`, 5-D three-body phase
  space, final PDGs `[23, 2, -2]`, LHC cuts (`pt_j_min=20`, `eta_j_max=2.5`,
  `dr_jj_min=0.4`, `m_jj_min=20`), condition range `[200, 2000]` GeV, held-out
  energies `[250, 400, 650, 1000, 1500, 1950]`.
- `generated/gg_zjj_standalone/`: MadGraph standalone build. Source/Cards are
  tracked; compiled objects and `matrix2py.cp310-win_amd64.pyd` are gitignored
  (matched by `*.py[cod]`). Rebuild with `madgraph_windows_build.py` if the
  `.pyd` is missing.
- `madgraph_process_integrand.py`:
  - `ThreeBodyPhaseSpace` now accepts `m23_min` and applies it in the
    phase-space map (`q23_low = m23_min**2`), so `m_jj_min` is not just a
    post-hoc rejection.
  - `ConfiguredMadGraphIntegrand._passes_cuts` applies `pt`, `|eta|` and
    `ΔR_jj`; `log_integrand` only calls the Fortran ME for passing points and
    sets rejected points to `-inf`.
- `experiments/validate_madgraph_process_integrand.py`: prints the
  cut-rejected fraction; fails only on non-finite momenta, a non-positive
  Jacobian, or zero surviving points.

### Equal-budget comparison

`experiments/validate_madgraph_process_integral.py` now supports:

- `--no-sobol`: skip the randomized-Sobol reference.
- `--compare-uniform`: draw exactly `--nf-evals` uniform points
  (`log q = 0`), evaluate the same integrand, and report estimate, stderr,
  relative stderr, ESS, time, and weight diagnostics.
- Per-energy diagnostics: `max_weight_over_mean`, `cut_pass_fraction`,
  `zero_weight_fraction`, `std_logw_finite`.
- Pull, `ESS_NF / ESS_uniform`, and variance reduction
  `(σ_uniform / σ_NF)^2`.
- UTF-8 stdout so the `σ` symbols print correctly on Windows.

Note: `cut_pass_fraction` is the **finite-weight fraction** — cut-rejected
points plus any point with a zero matrix element. For `gg_zjj` that is
dominated by the cuts, but do not describe it as a pure cut-efficiency number.

### Smoke evidence (not the final result)

- Smoke training: `results/gg_zjj_copula_gpu/` — 10 edges, ESS `0.492759`,
  integral `4.393305e-10 GeV^-2`. Checkpoint is present locally but ignored by
  git.
- Smoke NF-only validation at 100k evals:
  `results/gg_zjj_validation_gpu/summary.csv`.

| √ŝ (GeV) | NF estimate (pb) | stderr (pb) | rel. stderr | ESS | time (s) |
|---:|---:|---:|---:|---:|---:|
| 250 | 0.29113125 | 0.00083974 | 0.288% | 0.5459 | 6.45 |
| 400 | 0.36181907 | 0.00086825 | 0.240% | 0.6346 | 6.68 |
| 650 | 0.25153029 | 0.00058066 | 0.231% | 0.6523 | 6.77 |
| 1000 | 0.15564585 | 0.00034614 | 0.222% | 0.6691 | 6.87 |
| 1500 | 0.09317988 | 0.00020325 | 0.218% | 0.6776 | 6.83 |
| 1950 | 0.06615869 | 0.00015442 | 0.233% | 0.6473 | 6.80 |

- A 5,000-point `--compare-uniform` preview on the **smoke** checkpoint gave
  ESS ratios of 2.1–12.5× and variance reductions of 3.4–32.6× (e.g. at
  1950 GeV: ESS 0.612 vs 0.049, 32.6×). This was a temporary check in
  `/tmp/gg_zjj_compare_smoke`, not a committed result and not the balanced
  model. Re-run it on the balanced checkpoint as the real result.

### High-dimensional investigation (separate workstream)

`docs/HIGH_DIM_FINDINGS.md` records the 32-D reproduction. In short: the model
code is dimension-generic; the smoke annealing schedule collapses the weighted
cache, the smoke cache is too small for 32-D, and the per-dimension 40-step
bisection is the scaling wall. Fixes in
`experiments/run_adaptive_integrand_structure_search.py`:

- smoke profile uses `HIGH_DIM_MARGINAL_TEMPERATURES` when `DIM >= 32`;
- `--inverse-iterations` (default 40; 8–12 for high-dimensional training);
- `--temperatures` schedule override;
- dtype-correct uniform draws in the proposal and copula samplers;
- `std_logw` computed over finite weights.

The 32-D `thorough` run is **not** complete; that remains the verification step.

### Visualizer and documents

- `web_visualizer/` + `scripts/dump_web_telemetry.py`: the static 4-panel
  debugger for the **`gg_ttbar`** smoke model (not `gg_zjj`). Open
  `web_visualizer/index.html` directly or serve the folder; regeneration
  instructions are in `web_visualizer/README.md`. The telemetry is ~7.6 MB and
  is committed so the page works without running torch/MadGraph.
- `docs/VISUALIZATION_SPEC.md`: the design spec for that visualizer.
- `scripts/visualize_conditional_marginal.py` and
  `results/figures/conditional_marginal_anatomy.{png,svg}`: the four-panel
  marginal anatomy figure from the thorough `gg_ttbar` checkpoint.

## 3. Remaining task — 200k comparison

Run this **only after** the final checkpoint and summary exist:

```bash
cd "$REPO/generated/gg_zjj_standalone"
PYTHONPATH="$REPO:$REPO/experiments" "$PY" -u \
  "$REPO/experiments/validate_madgraph_process_integral.py" \
  --config "$REPO/configs/processes/gg_zjj.json" \
  --checkpoint "$REPO/results/gg_zjj_copula_balanced_gpu/madgraph_gg_zjj_5_final_model.pt" \
  --nf-evals 200000 \
  --gpu-chunk 20000 \
  --no-sobol \
  --compare-uniform \
  --device cuda \
  --output "$REPO/results/gg_zjj_validation_balanced_gpu"
```

Expected runtime: **~2–3 minutes** for all six energies. Basis: the 100k NF
smoke run took 6.45–6.87 s per energy, so 200k is ~12–14 s; the 5k uniform run
took 0.16–0.21 s per energy, so 200k is ~6–8 s. Per energy ~20 s, times six,
plus checkpoint load.

Outputs in `results/gg_zjj_validation_balanced_gpu/`:
`replicate_results.csv`, `summary.csv`, `config.json`. Both the NF and
`uniform_mc` rows are written. Print the full six-energy table with estimate,
stderr, relative error, ESS, time, pull, ESS improvement, variance reduction,
and the weight diagnostics.

## 4. Verification performed

- `pytest -q` → **12 passed**.
- `python -m py_compile` on every modified/new Python file → OK.
- `node --check web_visualizer/app.js` → OK.
- `git diff --check` → clean.
- Smoke `gg_zjj` training, cut-aware integrand validation, 100k NF validation,
  and a 5k `--compare-uniform` run all completed.

## 5. Gotchas

- `*.pt` checkpoints are gitignored. The smoke checkpoint and the eventual
  balanced checkpoint are local-only; a fresh clone cannot rerun validation
  without retraining or copying them.
- The balanced output directory does not exist until the run finishes; this is
  expected.
- The background task IDs from the previous Claude session are session-local.
  Use the log, the Windows PID, and the filesystem instead.
- `uniform_estimate` evaluates all 200k uniform points in one
  `log_integrand` call. The Fortran wrapper is a Python per-event loop, so that
  is CPU work (~7 s), not a GPU operation; memory is small.
- The web telemetry `meta` block embeds absolute paths from this machine.
  Regenerating on another machine is expected to rewrite them.
- Do not judge 32-D accuracy with `--speed-profile smoke`; see
  `docs/HIGH_DIM_FINDINGS.md`.
- The current balanced run was started **without** `--inverse-iterations`, so it
  uses the 40-step evaluation default. That is intentional for this production
  run; the knob is for future high-dimensional training.

## 6. Commit contents

This handoff is committed with:

- modified: `madgraph_process_integrand.py`,
  `experiments/run_adaptive_integrand_structure_search.py`,
  `experiments/run_bayesian_selected_copula_benchmark.py`,
  `experiments/run_copula_bin_sweep.py`,
  `experiments/validate_madgraph_process_integral.py`,
  `experiments/validate_madgraph_process_integrand.py`;
- new: `configs/processes/gg_zjj.json`, `generated/gg_zjj_standalone/`,
  `results/gg_zjj_copula_gpu/` (CSV only), `results/gg_zjj_validation_gpu/`,
  `results/figures/`, `scripts/dump_web_telemetry.py`,
  `scripts/visualize_conditional_marginal.py`, `web_visualizer/`,
  `docs/HIGH_DIM_FINDINGS.md`, `docs/VISUALIZATION_SPEC.md`, this file, and a
  pointer at the top of `CLAUDE.md`.

Ignored and therefore **not** committed: the `.pt` checkpoints, the compiled
`.pyd`/`.o`/`.mod` files, logs, and `__pycache__`.
