# Reproduction commands — independent Windows rebuild of the scattering sweep

Every command below was run on this machine, in this order, to produce the
results in `results/`. The point of the document is a like-for-like comparison
against `results_public/`: same process cards, same phase-space code, same QMC
protocol — so any disagreement in the numbers is attributable to something
identifiable rather than to an unknown difference in setup.

If the Sobol column reproduces exactly (it does — see §7), then the generated
matrix element, particle ordering, phase-space Jacobian, α_s and QMC protocol
all match. That is what this file is for checking.

---

## 1. Provenance

| item | value |
|---|---|
| repository state | `a8dad44` ("commit all"), branch `main` |
| machine | Windows 11, 16 logical cores, 15 GB RAM, **no GPU** |
| date | 2026-10-10 |
| MadGraph | MG5_aMC@NLO **3.6.7** (build date 2026-01-05) |
| Fortran/C toolchain | MSYS2 MinGW-w64 GCC/GFortran **14.1.0** |
| CPython | **3.10.14** (conda env `workspace`, MSVC build) |
| numpy | 1.26.4 |
| scipy | 1.15.3 |
| pandas | 2.2.2 |
| torch | 2.3.0+cpu |
| f2py | from numpy 1.26.4 |

---

## 2. The command card sent to MadGraph

Not hand-written. `experiments/setup_madgraph_process.py` constructs it from
`configs/processes/<name>.json` (unmodified, committed files):

```
set automatic_html_opening False
import model <spec["model"]>
generate <spec["process"]>
output standalone <absolute output path> --prefix=int
```

which for the three processes resolves to:

| config | `import model` | `generate` |
|---|---|---|
| `gg_ttbar.json` | `sm` | `g g > t t~ @1` |
| `gg_tth.json` | `sm` | `g g > t t~ h @1` |
| `uu_ww_4l.json` | `sm` | `u u~ > w+ w- , w+ > e+ ve, w- > mu- vm~ @1` |

The only field supplied by the operator is the absolute output path.
`output standalone … --prefix=int` is the f2py mode, which is the mode that
produces the `matrix2py` module `MadGraphMatrixElement` loads.

**MG5's own record of the card it ran is committed**, at
`generated/<process>_standalone/Cards/proc_card_mg5.dat`. It contains the
full card including MG5's own defaults, so it can be **diffed directly**
against the same file in your build trees:

```
set group_subprocesses Auto
set ignore_six_quark_processes False
set low_mem_multicore_nlo_generation False
set complex_mass_scheme False
set include_lepton_initiated_processes False
set gauge unitary
set loop_optimized_output True
set loop_color_flows False
set max_npoint_for_channel 0
set default_unset_couplings 99
set max_t_for_channel 99
set zerowidth_tchannel True
set nlo_mixed_expansion True
set automatic_html_opening False
import model sm
define p = g u c d s u~ c~ d~ s~
define j = g u c d s u~ c~ d~ s~
define l+ = e+ mu+
define l- = e- mu-
define vl = ve vm vt
define vl~ = ve~ vm~ vt~
generate g g > t t~ @1                       # gg_tth: g g > t t~ h @1
                                             # uu_ww_4l: u u~ > w+ w- , w+ > e+ ve, w- > mu- vm~
output standalone <absolute path> --prefix=int
```

Every `set …` line above is an MG5 3.6.7 default, not an operator choice; the
banner records `VERSION 3.6.7  2026-01-05`. So the card is fully determined by
the committed config plus the MG5 defaults, and any other 3.6.7 user running
the same `generate` line gets the same setup.

---

## 3. Commands, in order

`$REPO` = `C:/Users/laiji/test/conditional-nf-integration`
`$PY`   = the conda 3.10 interpreter

### 3.1 Generate and compile a standalone (once per process)

```bash
cd "$REPO"
PYTHONPATH="$REPO" $PY experiments/setup_madgraph_process.py \
    --config configs/processes/gg_ttbar.json \
    --mg5 /path/to/mg5_aMC
```

(repeat for `gg_tth.json`, `uu_ww_4l.json`)

### 3.2 Integrand sanity check

> Must run with the **CWD inside the standalone directory** — `lha_read.f`
> resolves `./Cards/` relative to the CWD and calls a Fortran `stop` (exit
> status 0, no message) if it fails.

```bash
cd "$REPO/generated/gg_ttbar_standalone"
PYTHONPATH="$REPO" $PY experiments/validate_madgraph_process_integrand.py \
    --config "$REPO/configs/processes/gg_ttbar.json" --points 500
```

### 3.3 Train

```bash
cd "$REPO/generated/gg_ttbar_standalone"
PYTHONPATH="$REPO" $PY -u experiments/run_madgraph_process_conditional_copula.py \
    --config "$REPO/configs/processes/gg_ttbar.json" \
    --speed-profile smoke \
    --no-compile-kernels \
    --output "$REPO/results/gg_ttbar_copula_smoke"
```

### 3.4 Held-out validation

```bash
cd "$REPO/generated/gg_ttbar_standalone"
PYTHONPATH="$REPO" $PY -u experiments/validate_madgraph_process_integral.py \
    --config "$REPO/configs/processes/gg_ttbar.json" \
    --checkpoint "$REPO/results/gg_ttbar_copula_smoke/madgraph_gg_ttbar_2_final_model.pt" \
    --output "$REPO/results/gg_ttbar_validation_smoke"
```

All validation options are the driver defaults, which match the committed
`results_public/*/config.json`:

```
sobol_power = 18      sobol_replicates = 8      nf_evals = 500000
uniform_fraction = 0.1      alpha_s = 0.118      seed = 20260920
```

Energies are `condition.held_out` from each config. The one settings difference
is `--device`: `cpu` here, `cuda` in the committed configs.

---

## 4. The one deviation: how the wrapper was compiled

`compile_libraries()` shells out to MadGraph's makefiles, which are POSIX-only
(`SHELL = /bin/bash`, `uname -s`, `-Wl,-rpath`, `.so` suffixes) and produce no
importable extension on Windows. The replacement builds the same f2py extension
directly — same `f2py_wrapper.f` + `matrix.f`, linked against the same
`lib/libdhelas.a` and `lib/libmodel.a` that MadGraph's own make builds during
generation:

```bash
cd "$REPO/generated/gg_ttbar_standalone/SubProcesses/P1_gg_ttx"
cp ../../Source/MODEL/coupl.inc .        # MG5 symlinks this; Windows cannot
LDFLAGS="-static" $PY -m numpy.f2py -c \
    f2py_matrix_wrapper.f matrix.f \
    -L"$REPO/generated/gg_ttbar_standalone/lib" -ldhelas -lmodel \
    -m matrix2py \
    --compiler=mingw32 --fcompiler=gnu95 \
    --f77flags="-ffixed-line-length-132 -w"
```

Two flags matter:

* `--compiler=mingw32 --fcompiler=gnu95` — f2py then generates the
  `libpython310.a` import library it needs from `python310.dll`, which is what
  lets a MinGW-built extension be imported by MSVC-built CPython.
* `LDFLAGS=-static` — without it the extension depends on
  `libgfortran-5.dll` / `libgcc_s_seh-1.dll` / `libquadmath-0.dll` /
  `libwinpthread-1.dll`; CPython 3.8+ does not search `PATH` for an extension's
  dependencies, so the import fails.

**This is the only substantive difference from an MG5-makefile build**, and it
changes the compiler optimiser flags applied to `matrix.f` (f2py supplies
`-O3 -funroll-loops`). That is the likely origin of the two last-digit
differences seen in §7 for the 8-D process.

---

## 5. What was NOT changed

* `configs/processes/*.json` — untouched.
* the phase-space maps in `madgraph_process_integrand.py` /
  `madgraph_tth_integrand.py` — untouched.
* the training engine `run_adaptive_integrand_structure_search.py` — untouched.
* the QMC protocol in `validate_madgraph_process_integral.py` — untouched.

Two small edits were required for Windows only (both no-ops on Linux):
`MadGraphMatrixElement` now globs `matrix2py*.pyd` as well as `*.so`, and
`compile_libraries()` delegates to the MinGW builder when `sys.platform ==
"win32"`.

---

## 6. Integrand validation output

`validate_madgraph_process_integrand.py --points 300..500`:

| quantity | gg_ttbar (2D) | gg_tth (5D) | uu_ww_4l (8D) |
|---|---:|---:|---:|
| max 4-momentum residual | 2.27373675443e-13 | 4.54747350886e-13 | 4.54747350886e-13 |
| max mass-shell residual | 3.49245965481e-10 | 5.23868948221e-10 | 3.49245965481e-10 |
| minimum Jacobian | 0.0115863199095 | 0.0481909697582 | 0.404088762337 |
| finite fraction | 1 | 1 | 1 |
| positive integrand fraction | 1 | 1 | 1 |
| throughput [points/s] | ~97 000 | ~53 000 | ~143 000 |

For reference, the same check on Linux for `gg_ttbar` gave
`2.27373675443e-13`, `3.49245965481e-10`, `0.0115863199095` and
47 203 points/s — every printed digit of the first three matches.

---

## 7. Comparison against `results_public/`

**Training budget, stated once and applying to every table below.** The runs in
`results/` used `--speed-profile smoke` (300 KL steps). The published runs in
`results_public/` used `--speed-profile thorough` (30 000 KL steps) on CUDA.
Every row below is labelled `smoke` or `thorough` so the two are not confused;
where a table has no label, the column header carries it.

The Sobol column (§7.1) is **not** affected by this: it is a property of the
generated matrix element, the phase-space map and the QMC protocol, none of
which the training budget touches. That is why it is the meaningful
build-level check.

### 7.1 Sobol column — the build check

Compared at the **per-replicate** level (8 replicates × 5 energies = 40 rows
per process), not just the summary means:

| process | reference dir | replicates bit-identical | worst deviation |
|---|---|---:|---|
| gg → t t̄ | `gg_ttbar_validation` | **40 / 40** | 0 |
| gg → t t̄ H | `gg_tth_5d_validation` | **40 / 40** | 0 |
| u ū → 4ℓ | `uu_ww_4l_validation` | 38 / 40 | 1.0e-16 abs (6.9e-15 rel) |

The two differing `uu_ww_4l` replicates are at 800 GeV (replicate 1) and
differ in the 16th significant digit — consistent with a different optimiser
applied to the 8-D matrix element, and below the precision of the published
summary. At the summary level all 15 values are identical to every printed
digit.

### 7.2 Training summaries — `*_final_summary.csv`

`smoke` (here) vs `thorough` (published), the file each training run writes at
the end of its budget:

| process | dim | edges smoke / thorough | ESS smoke | ESS thorough | ratio | Î smoke | Î thorough | ratio |
|---|---:|:---:|---:|---:|:---:|---:|---:|:---:|
| gg → t t̄ | 2 | 1 / 1 | 0.8054 | 0.8406 | 0.96 | 2.632163e-08 | 2.635498e-08 | 0.99873 |
| gg → t t̄ H | 5 | 10 / 10 | 0.8159 | 0.9176 | 0.89 | 9.646322e-11 | 9.697110e-11 | 0.99476 |
| u ū → 4ℓ | 8 | 28 / 28 | 0.4535 | 0.5078 | 0.89 | 3.697135e-11 | 3.735932e-11 | 0.98962 |

| process | log w std smoke | log w std thorough |
|---|---:|---:|
| gg → t t̄ | 0.498 | 0.453 |
| gg → t t̄ H | 0.700 | 0.595 |
| u ū → 4ℓ | 1.159 | 1.023 |

Three things to read off this table:

* the **edge count is identical in all three cases** — the graph-selection step
  lands on the same structure from 300 KL steps as from 30 000, which is a
  structural agreement independent of budget;
* the **integral estimates agree to 0.1–1 %**;
* ESS is uniformly ~10 % lower for `smoke`, with the log-weight spread
  correspondingly ~10–13 % wider.

### 7.3 Held-out NF column — **not** a reproduction, by design

`smoke` models (here) against the published `thorough` models, on fresh samples
at held-out energies:

| process | ESS/N smoke | ESS/N thorough | max pull |
|---|---:|---:|---:|
| gg → t t̄ | 0.812 – 0.980 | 0.981 – 0.998 | 0.89σ |
| gg → t t̄ H | 0.797 – 0.917 | 0.959 – 0.987 | 1.54σ |
| u ū → 4ℓ | 0.806 – 0.865 | 0.927 – 0.941 | 0.42σ |

### 7.4 Two different "ESS" numbers per model

Worth separating, because the two diagnostics differ by a lot — and they differ
for the **published** models as much as for these. Both columns below are the
authors' own data, read from `results_public/`:

| process | training-time ESS (thorough, published) | validated ESS (thorough, published) |
|---|---:|---:|
| gg → t t̄ | 0.8406 | 0.981 – 0.998 |
| gg → t t̄ H | 0.9176 | 0.959 – 0.987 |
| u ū → 4ℓ | **0.5078** | **0.927 – 0.941** |

For the 8-D process the `final_summary.csv` figure of 0.51 sits against a model
that actually delivers 0.93 on held-out samples. Both are legitimate
measurements of different things, but `final_summary.csv` is the file a reader
opens first.

A practical consequence for reading §7.2: the training-time ESS is the
pessimistic diagnostic on both sides, so the smoke-vs-thorough gap there
(0.4535 vs 0.5078 for 8-D) is much narrower than the gap in actual held-out
performance (§7.3: 0.806–0.865 vs 0.927–0.941).

---

## 8. Three things that may be worth a look

Offered as observations from running the code, not as criticisms.

1. **`compile_libraries()`'s combined-target branch is unreachable.**
   `setup_madgraph_tth_benchmark.py` runs `make allmatrix2py.so`, but
   `makefile_sa_f2py` defines the target as `all_matrix2py.so` — with an
   underscore (`f2py … -m all_matrix$(MENUM)py`). The names never match, so the
   per-process fallback is always taken and the printed message
   *"Combined allmatrix2py target is unavailable"* misattributes this to the
   MG5 build.

2. **No trained checkpoints are committed.** There are no `.pt` files anywhere
   under `results_public/`; `gg_ttbar_thorough_unbuffered/` ships the summary
   CSV and history only. That makes the NF column of every published table
   unverifiable without retraining from scratch. (Note it still would not
   reproduce bit-for-bit even with the checkpoint, because torch's CPU and CUDA
   RNGs differ — so the Sobol column is hardware-independent and the NF column
   is not. The tables don't currently distinguish them.)

3. **`final_summary.csv`'s `ess` reads much lower than the validated ESS**, for
   the published models as much as for these. The numbers are in §7.4; the
   point is that for the 8-D process the summary figure (0.5078) is less than
   half the model's actual held-out performance (0.927 – 0.941), and
   `final_summary.csv` is the file a reader opens first.
