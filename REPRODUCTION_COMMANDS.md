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

## 3. Windows — setup and individual commands

### 3.1 One-time setup

**Shell variables used throughout this section:**

```bash
export REPO="C:/Users/laiji/test/conditional-nf-integration"
export PY="/c/Users/laiji/.conda/envs/workspace/python.exe"
export MG5="C:/Users/laiji/test/bin/mg5_aMC.bat"
```

Note the two path styles: `$REPO` and `$MG5` are **Windows-style** because they
are passed as arguments to `python.exe`, which does not understand `/c/...`;
`$PY` is POSIX-style because Git Bash has to *execute* it.

**(a) Python environment** — conda env `workspace`, Python 3.10.14:

```bash
conda activate workspace
pip install "numpy<2" pytest
pip install --only-binary :all: vegas
```

`numpy<2` is required **only** because this env carries torch 2.3.0, which is
compiled against NumPy 1.x. With a newer torch the pin is unnecessary — see
`requirements.txt`.

**(b) MinGW Fortran toolchain** — from a MSYS2 shell:

```bash
pacman -S --needed mingw-w64-x86_64-gcc-fortran
```

**(c) MadGraph 3.6.7:**

```bash
cd /c/Users/laiji/test
tar xzf /c/Users/laiji/Downloads/MG5_aMC_v3.6.7.tar.gz
```

`tar` exits 2 and reports three failed symlinks (`gpu/rambo.cc`,
`pineappl_maxproc.inc`, `NNPDFDriver.f`). All three belong to GPU/NLO/PDF
features this repository never touches; the tree is usable.

**(d) MG5 launcher** — `bin/mg5_aMC` is a POSIX script with a
`#!/usr/bin/env python3` shebang that Windows `CreateProcess` cannot execute,
so `setup_madgraph_process.py` (which does `subprocess.run([mg5, card])`) needs
a `.bat`. Create `C:\Users\laiji\test\bin\mg5_aMC.bat`:

```bat
@echo off
set "PATH=C:\msys64\usr\bin;C:\msys64\mingw64\bin;%PATH%"
"C:\Users\laiji\.conda\envs\workspace\python.exe" "C:\Users\laiji\test\MG5_aMC_v3_6_7\bin\mg5_aMC" %*
```

MSYS2 goes on `PATH` because MG5 hard-requires `gzip` at startup.

**(e) CUDA-enabled torch, for GPU runs only.** The conda env ships
`torch 2.3.0+cpu` — the CPU-only build — so `torch.cuda.is_available()` returns
`False` and every run silently falls back to CPU:

```bash
"$PY" -m pip install "torch==2.3.0+cu121" \
    --index-url https://download.pytorch.org/whl/cu121
```

`+cu121` and not `+cu124` because the installed driver (546.92) advertises CUDA
12.3, and CUDA minor-version compatibility makes cu121 the safe match. The torch
*version* is held at 2.3.0 so the `numpy<2` pin stays valid.

Verify before running anything:

```bash
"$PY" -c "
import torch
print('torch          ', torch.__version__)
print('cuda available ', torch.cuda.is_available())
print('device         ', torch.cuda.get_device_name(0) if torch.cuda.is_available() else '-')
free, total = torch.cuda.mem_get_info()
print('vram free/total: %.2f / %.2f GiB' % (free/2**30, total/2**30))
"
```

To revert to CPU: same command with `torch==2.3.0+cpu` and
`--index-url https://download.pytorch.org/whl/cpu`.

### 3.2 Generate and compile a standalone (once per process)

```bash
cd "$REPO"
PYTHONPATH="$REPO" $PY experiments/setup_madgraph_process.py \
    --config configs/processes/gg_ttbar.json \
    --mg5 /path/to/mg5_aMC
```

(repeat for `gg_tth.json`, `uu_ww_4l.json`)

### 3.3 Integrand sanity check

> Must run with the **CWD inside the standalone directory** — `lha_read.f`
> resolves `./Cards/` relative to the CWD and calls a Fortran `stop` (exit
> status 0, no message) if it fails.

```bash
cd "$REPO/generated/gg_ttbar_standalone"
PYTHONPATH="$REPO" $PY experiments/validate_madgraph_process_integrand.py \
    --config "$REPO/configs/processes/gg_ttbar.json" --points 500
```

### 3.4 Train

**CPU:**

```bash
cd "$REPO/generated/gg_ttbar_standalone"
PYTHONPATH="$REPO" $PY -u experiments/run_madgraph_process_conditional_copula.py \
    --config "$REPO/configs/processes/gg_ttbar.json" \
    --speed-profile smoke \
    --no-compile-kernels \
    --output "$REPO/results/gg_ttbar_copula_smoke"
```

**GPU** — three differences from the CPU form:

```bash
cd "$REPO/generated/gg_ttbar_standalone"
PYTHONPATH="$REPO" $PY -u experiments/run_madgraph_process_conditional_copula.py \
    --config "$REPO/configs/processes/gg_ttbar.json" \
    --speed-profile smoke \
    --device cuda \
    --output "$REPO/results/gg_ttbar_copula_smoke_gpu"
```

1. `--device cuda` — pass it explicitly; the default is decided at import time
   by `torch.cuda.is_available()`, so a silent CPU fallback is otherwise easy.
2. **Drop `--no-compile-kernels`.** That flag exists to *disable* the CUDA
   kernel-compile path; on a GPU you want it on.
3. Write to a `_gpu`-suffixed output directory so the CPU and GPU runs can be
   compared rather than silently overwriting each other.

> **`--output` is mandatory in both forms.** Omit it and the driver falls back
> to the relative default `results/<name>_conditional_copula`, which resolves
> against the CWD — i.e. inside `generated/<proc>_standalone/`, where nothing
> looks for it. This is a silent misplacement, not an error.

### 3.5 Held-out validation

**CPU:**

```bash
cd "$REPO/generated/gg_ttbar_standalone"
PYTHONPATH="$REPO" $PY -u experiments/validate_madgraph_process_integral.py \
    --config "$REPO/configs/processes/gg_ttbar.json" \
    --checkpoint "$REPO/results/gg_ttbar_copula_smoke/madgraph_gg_ttbar_2_final_model.pt" \
    --output "$REPO/results/gg_ttbar_validation_smoke"
```

**GPU:** add `--device cuda` and point at the `_gpu` checkpoint.

```bash
cd "$REPO/generated/gg_ttbar_standalone"
PYTHONPATH="$REPO" $PY -u experiments/validate_madgraph_process_integral.py \
    --config "$REPO/configs/processes/gg_ttbar.json" \
    --device cuda \
    --checkpoint "$REPO/results/gg_ttbar_copula_smoke_gpu/madgraph_gg_ttbar_2_final_model.pt" \
    --output "$REPO/results/gg_ttbar_validation_smoke_gpu"
```

All validation options are the driver defaults, which match the committed
`results_public/*/config.json`:

```
sobol_power = 18      sobol_replicates = 8      nf_evals = 500000
uniform_fraction = 0.1      alpha_s = 0.118      seed = 20260920
```

Energies are `condition.held_out` from each config. The one settings difference
against the published configs is `--device`: `cpu` here, `cuda` in the committed
configs.

> The **Sobol column is hardware-independent** and reproduces
> `results_public/` bit-for-bit on either device. The **NF column is not** —
> and would not reproduce bit-for-bit even with the authors' checkpoint, since
> torch's CPU and CUDA RNGs produce different streams from the same seed.

---

## 4. lxplus — setup and individual commands

### 4.1 What is already installed there

From `env.sh` in this repository (paths are specific to
`/eos/user/j/jlai/mango`):

| what | path |
|---|---|
| repo | `$NFI_ROOT/conditional-nf-integration` |
| Python | `$NFI_ROOT/venv/bin/python` (3.11.5) |
| MadGraph | `$NFI_ROOT/MG5_aMC_v3_6_7`, invoked through `$NFI_ROOT/bin/mg5_aMC` |
| standalone | `$NFI_REPO/generated/gg_ttbar_standalone` |

```bash
source /eos/user/j/jlai/mango/conditional-nf-integration/env.sh
```

That exports `NFI_REPO`, `NFI_PY`, `NFI_MG5`, `NFI_STANDALONE` and `PYTHONPATH`.
Unlike Windows, `bin/mg5_aMC` is directly executable there — no `.bat` wrapper
is needed, and the MinGW/f2py workaround in §5 does not apply.

### 4.2 Check what GPU you actually have before planning around it

```bash
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv
"$NFI_PY" -c "import torch; print(torch.__version__, torch.cuda.is_available())"
```

The lxplus environment recorded in `PRD.md` Part II had **no GPU**, and its venv
carries `torch 2.14.1+cpu` — the CPU-only build. So on that machine both checks
are expected to fail. GPU work needs a GPU node *and* a CUDA-enabled torch.

**Which GPU resource to use is not settled here.** CERN provides GPUs through
several routes (HTCondor GPU queues, dedicated pools), and the correct one
depends on your account and the current batch configuration. Confirm the queue
name and the `request_gpus` syntax with the site before submitting — do not
assume the values below.

### 4.3 Making torch see a GPU

Only needed on a node that has one. Install the CUDA build of the **same torch
version** the venv already has, so the `numpy<2` pin stays valid:

```bash
"$NFI_PY" -m pip install "torch==2.14.1" \
    --index-url https://download.pytorch.org/whl/cu121
```

Match the `cuXXX` tag to the node's driver. Verify with the same check as §4.2.

### 4.4 Commands

Identical in structure to §3.2–§3.5 on Windows, with two substitutions: use
`$NFI_PY` and `$NFI_REPO`, and drop the Windows-only bits. The **working
directory rule is unchanged and still mandatory** — `lha_read.f` resolves
`./Cards/` against the CWD and calls a Fortran `stop` (exit status 0, no
output) if it fails.

```bash
export REPO="$NFI_REPO"
export PY="$NFI_PY"

# generate + compile (once per process)
cd "$REPO"
PYTHONPATH="$REPO" $PY experiments/setup_madgraph_process.py \
    --config configs/processes/gg_ttbar.json --mg5 "$NFI_MG5"
```

Then, on a node with a GPU and a CUDA-enabled torch (§4.2, §4.3):

```bash
# --- everything below runs with CWD inside the standalone ---
cd "$REPO/generated/gg_ttbar_standalone"

# integrand check — CPU-side, does not touch the flow
PYTHONPATH="$REPO" $PY experiments/validate_madgraph_process_integrand.py \
    --config "$REPO/configs/processes/gg_ttbar.json" --points 500

# train, smoke, GPU
PYTHONPATH="$REPO" $PY -u experiments/run_madgraph_process_conditional_copula.py \
    --config "$REPO/configs/processes/gg_ttbar.json" \
    --speed-profile smoke --device cuda \
    --output "$REPO/results/gg_ttbar_copula_smoke_gpu"

# held-out validation, GPU
PYTHONPATH="$REPO" $PY -u experiments/validate_madgraph_process_integral.py \
    --config "$REPO/configs/processes/gg_ttbar.json" \
    --device cuda \
    --checkpoint "$REPO/results/gg_ttbar_copula_smoke_gpu/madgraph_gg_ttbar_2_final_model.pt" \
    --output "$REPO/results/gg_ttbar_validation_smoke_gpu"
```

Two differences from the CPU block: `--device cuda` is passed explicitly (the
default is decided at import time by `torch.cuda.is_available()`), and
`--no-compile-kernels` is dropped so the deterministic integrand kernel can
compile. `--output` is mandatory here too (see the note in §3.4).

For thorough, change one flag and the output name:

```bash
PYTHONPATH="$REPO" $PY -u experiments/run_madgraph_process_conditional_copula.py \
    --config "$REPO/configs/processes/gg_ttbar.json" \
    --speed-profile thorough --device cuda \
    --output "$REPO/results/gg_ttbar_copula_thorough_gpu"
```

**Measured durations** for `--speed-profile thorough --device cuda` on an
RTX 3060 Laptop GPU, for sizing a batch request:

| process | dim | wall clock |
|---|---:|---:|
| `gg → t t̄` | 2 | **5 h 05 min** |
| `gg → t t̄ H` | 5 | **12 h 26 min** |
| `u ū → 4ℓ` | 8 | **13 h 15 min** |

These are dominated by the single-threaded CPU matrix-element loop (§9.2), not
by the GPU. A lxplus GPU node will only beat them if its CPU is faster.

### 4.5 Long runs must go through the batch system

Do **not** run a `thorough` job interactively on lxplus. It holds one CPU core
for many hours, and the login nodes enforce CPU-usage limits — that is exactly
what forced this work onto a local machine. `CLAUDE.md` in this repository
documents the HTCondor setup: `run_batch.sh` plus `sweeps.sub`, submitted with
the EosSubmit schedd loaded (`module load lxbatch/eossubmit`), every referenced
file on `/eos`, and one log file per cluster.

**Wall time is the thing that will bite you.** `+JobFlavour` is a hard maximum,
not a reservation, and a thorough run is **all-or-nothing** — the driver writes
its checkpoint only at the very end, so a kill at the wall time loses the entire
process. Against the measured durations in §4.4:

| process | measured | minimum flavour |
|---|---:|---|
| 2→2 `gg→tt̄` | 5 h 05 min | `workday` (8 h) fits |
| 2→3 `gg→tt̄H` | 12 h 26 min | **`tomorrow` (24 h)** — `workday` kills it at 8 h |
| 2→4 `uū→4ℓ` | 13 h 15 min | **`tomorrow` (24 h)** |

This is not hypothetical: `CLAUDE.md` §3.4 records a previous submission killed
by a `longlunch` (2 h) limit.

The existing `sweeps.sub` requests `request_cpus = 2`, **no GPU**, and queues
only smoke budgets. A GPU thorough run needs a `request_gpus` line and a
GPU-capable queue; confirm both with the site before submitting (§4.2).

### 4.6 The honest expectation for GPU on lxplus

**A GPU there will not make this faster.** The runtime is set by the matrix
element, which is evaluated single-threaded on the CPU — see §9. Measured
directly: moving the flow from CPU to an RTX 3060 changed a smoke run from
303 s to 304 s.

What lxplus actually offers is **cores**. The per-event matrix-element calls are
independent, so a multiprocessing pool around `log_integrand` should give close
to linear speedup, and that is a far better use of the machine than a GPU.

---

## 5. The one deviation: how the wrapper was compiled

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

## 6. What was NOT changed

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

## 7. Integrand validation output

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

## 8. Comparison against `results_public/`

**Training budget, stated once and applying to every table below.** The runs in
`results/` used `--speed-profile smoke` (300 KL steps). The published runs in
`results_public/` used `--speed-profile thorough` (30 000 KL steps) on CUDA.
Every row below is labelled `smoke` or `thorough` so the two are not confused;
where a table has no label, the column header carries it.

The Sobol column (§7.1) is **not** affected by this: it is a property of the
generated matrix element, the phase-space map and the QMC protocol, none of
which the training budget touches. That is why it is the meaningful
build-level check.

### 8.1 Sobol column — the build check

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

### 8.2 Training summaries — `*_final_summary.csv`

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

### 8.3 Held-out NF column — **not** a reproduction, by design

`smoke` models (here) against the published `thorough` models, on fresh samples
at held-out energies:

| process | ESS/N smoke | ESS/N thorough | max pull |
|---|---:|---:|---:|
| gg → t t̄ | 0.812 – 0.980 | 0.981 – 0.998 | 0.89σ |
| gg → t t̄ H | 0.797 – 0.917 | 0.959 – 0.987 | 1.54σ |
| u ū → 4ℓ | 0.806 – 0.865 | 0.927 – 0.941 | 0.42σ |

### 8.4 Two different "ESS" numbers per model

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

## 9. GPU runs

### 9.1 Measured: the GPU buys nothing on this workload

`gg → t t̄`, `--speed-profile smoke`, identical config, identical budget; the
only difference is `--device`:

| | CPU | GPU (RTX 3060 Laptop) |
|---|---:|---:|
| training wall clock | **303 s** | **304 s** |
| validation wall clock | — | 208 s |
| learned graph | 1 edge | 1 edge |
| final ESS | 0.8054 | 0.8074 |
| final Î | 2.632163e-08 | 2.647052e-08 |

**0.3 %.** The models are statistically indistinguishable. The Sobol column of
the GPU validation is still bit-identical to `results_public/` at all five
energies, so the device does not affect the build-level check.

### 9.2 Why

**(a) The matrix element is the ceiling and it runs on the CPU.** It is
evaluated one event per Python call:

```python
for index, event in enumerate(momenta):
    values[index] = float(self.evaluate(p, self.alpha_s, -1))
```

The published `gg_ttbar` validation ran 500 000 evaluations in 5.34 s — about
**93 600/second** — *with the flow already on CUDA*. The matrix-element
throughput measured on the Windows box, on **one CPU core**, is
**96 600/second**. The two match, which says the matrix element dominates the
wall clock whether or not the flow is on a GPU.

**(b) The driver round-trips host↔device every chunk.** From
`run_madgraph_process_conditional_copula.py`:

```python
unit_numpy = unit.detach().cpu().double().numpy()          # GPU → host
for start in range(0, len(unit_numpy), me_chunk):
    pieces.append(evaluator.log_integrand(...))             # CPU matrix element
return torch.as_tensor(np.concatenate(pieces), device=unit.device, ...)  # host → GPU
```

Each chunk of `--me-chunk` (default 512) samples on the GPU, copies to the
host, evaluates the matrix element on one CPU core, and copies back.

### 9.3 Consequence for the Alpaka port

**Porting the flow to Alpaka cannot deliver a ≥10× throughput gate on this
architecture** — not because Alpaka is the wrong tool, but because the flow is
not where the time is. Three routes actually move the number:

1. **Remove the round trip** — co-resident sampling and evaluation. Small
   change, worth doing regardless.
2. **Move the matrix element to the GPU** — substantially harder than the flow,
   and not currently in the proposal's scope. This is where the real 10× lives.
3. **Benchmark at a multiplicity where the flow dominates** — high jet
   multiplicity, where the matrix element is relatively cheaper.

Because the gate is defined on total event-generation rate and the matrix
element is CPU-bound, no amount of Alpaka work on the flow alone will pass it.

**The cheapest real win is CPU-side:** the per-event matrix-element calls are
independent, so a multiprocessing pool around `log_integrand` should scale
close to linearly with cores. No Alpaka, no GPU.

### 9.4 Thorough runs — results

Three processes at `--speed-profile thorough --device cuda`, run sequentially on
an RTX 3060 Laptop GPU.

| process | status | wall clock | final ESS | Î |
|---|---|---|---:|---:|
| `gg → t t̄` | **complete** | 5 h 05 min | 0.840708 | 2.635616e-08 GeV⁻² |
| `gg → t t̄ H` | **complete** | 12 h 26 min | 0.918134 | 9.698626e-11 |
| `u ū → 4ℓ` | **complete** | 13 h 15 min | 0.508716 | 3.735308e-11 |

**2→2:** ESS 0.840708 against the published 0.840604 (rel. 1.2e-04). Î is
2.635616e-08 GeV⁻² (10.263 pb) against the published 2.635498e-08 — a relative
difference of 4.5e-05. This is the strongest model-level agreement in this
document: independent build, different Fortran toolchain, different operating
system, full training budget. Held-out validation gives pulls −1.41 / −0.73 /
+1.13 / +0.57 / +1.06 against Sobol with ESS 0.9820–0.9976, the **Sobol column
bit-identical** to `results_public/gg_ttbar_validation` and the NF column
agreeing to −0.017 %, −0.003 %, +0.033 %, +0.056 %, +0.023 %.

**2→3:** end-to-end validation at all five energies, pulls −0.32 / −0.53 /
−1.01 / +0.14 / +0.96 against Sobol, with ESS 0.9615–0.9836. The **Sobol column
is bit-identical** to `results_public/gg_tth_5d_validation`; the NF column
agrees to −0.004 %, +0.006 %, +0.001 %, +0.033 %, +0.067 %. Table in
`results/gg_tth_validation_thorough_gpu/summary.csv`.

**2→4:** pulls −2.23 / +0.63 / −0.29 / +0.35 / −0.46 against Sobol, ESS
0.9286–0.9434. The **Sobol column is bit-identical** to
`results_public/uu_ww_4l_validation`; the NF column agrees to +0.004 %,
+0.007 %, −0.001 %, −0.009 %, +0.011 %, and the ESS is slightly *higher* than
the published model at all five energies. Table in
`results/uu_ww_4l_validation_thorough_gpu/summary.csv`.

The −2.23 pull at 300 GeV is **not** a defect in this build: the published model
has the same pull (−2.3σ), so both independent models land in the same place and
the deviation is a property of the model at that energy.

One thing the 2→3 run exposes: the same 500 000 evaluations took **26.4–27.1 s
here against 11.2 s in `results_public`**, a factor of 2.4. That is the CPU
matrix-element loop, not the GPU — it is the axis the MANGO throughput gate is
defined on, and it means a faster CPU node is worth more than a faster GPU here.

**The training driver writes its checkpoint only when the run finishes**, and it
cannot resume from a saved file either — the checkpoint holds
`model_state_dict`, `marginal_state_dict`, `graph` and metrics, but no optimizer
state, RNG state or step position. The thorough runs are therefore
all-or-nothing: 5 h (2→2), 12.5 h (2→3), 13 h 15 min (2→4) of uninterrupted
wall clock. The 2→4 run lost 49 min and then 9 h 20 m to interruptions before
completing on its third attempt.

`run_thorough_gpu.sh` records the exact invocation. A corrected variant that
skips the completed process and fixes the `--output` handling is at
`mg5_scratch/run_thorough_gpu_resume.sh` — use the latter.

---

## 10. Four things that may be worth a look

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

4. **The χ² fine-tuning stage never pays for itself.** On all three processes,
   at the full `chi2_max_steps = 6000` budget:

   | process | ESS entering χ² | χ² best | outcome |
   |---|---:|---:|---|
   | 2→2 | 0.840873 | 0.840877 | 4,200 byte-identical steps, then −0.0007 |
   | 2→3 | 0.917502 | 0.917263 | −0.0002 |
   | 2→4 | 0.506192 | 0.506192 | **`best step = 0`** — the whole budget discarded |

   The 8-D case is unambiguous: `CHI2 SUMMARY` reports `best step = 0`, so the
   stage finished by restoring the weights it started with, after ~1 h of
   compute. Global ESS declined monotonically throughout (0.505483 → 0.505449)
   while local ESS sat at 0.99, i.e. the objective it optimises was already
   satisfied. Together with the never-approached `kl_max_steps = 30 000` (all
   three processes stop at 7 200 by patience), the thorough profile's budgets
   look substantially over-provisioned.
